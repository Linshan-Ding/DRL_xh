from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim

from agent.baselines.common import (
    A2CActor,
    CsvLogger,
    ValueNetwork,
    add_common_args,
    best_config_path,
    collect_metrics,
    collect_resource_config,
    init_base_env,
    initialize_episode_resources,
    load_orders,
    make_episode_env,
    make_visdom,
    max_decisions,
    normalized_to_env_action,
    output_paths,
    print_episode,
    print_config_source,
    save_checkpoint,
    set_seed,
    state_to_tensors,
    step_env,
    write_best_config_csv,
)


def build_parser():
    parser = argparse.ArgumentParser(description="A2C baseline for warehouse resource allocation.")
    add_common_args(parser)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda, device):
    returns = []
    advantages = []
    gae = torch.zeros(1, device=device)
    next_v = next_value.detach().view(1)
    values_detached = values.detach().view(-1)

    for step in reversed(range(len(rewards))):
        reward = torch.tensor([rewards[step]], dtype=torch.float32, device=device)
        done = torch.tensor([float(dones[step])], dtype=torch.float32, device=device)
        value = values_detached[step].view(1)
        delta = reward + gamma * next_v * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        advantages.insert(0, gae.squeeze(0))
        returns.insert(0, (gae + value).squeeze(0))
        next_v = value

    advantages = torch.stack(advantages)
    returns = torch.stack(returns)
    if advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def train(args):
    print_config_source()
    set_seed(args.seed)
    device = torch.device(args.device)

    orders = load_orders(args.items, args.month)
    base_env = init_base_env()
    decision_limit = max_decisions(args.mode, args.max_days)
    action_dim = base_env.N_a + 1
    scalar_dim = base_env.N_a + 1

    actor = A2CActor(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    value_net = ValueNetwork(base_env.N_w, base_env.N_l, scalar_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(value_net.parameters()), lr=args.lr)
    mse_loss = nn.MSELoss()

    csv_path, model_path = output_paths(args, "a2c")
    best_config_csv_path = best_config_path(args, "a2c")
    logger = CsvLogger(csv_path)
    viz, win = make_visdom(args, "a2c")
    best_cost = math.inf

    try:
        for episode in range(1, args.episodes + 1):
            env, state = make_episode_env(base_env, orders)
            log_probs, entropies, values, rewards, dones = [], [], [], [], []
            state, done, first_decision_index, episode_config_rows = initialize_episode_resources(
                env,
                args.mode,
                episode,
                args.items,
                args.month,
                args.seed,
                "a2c",
            )
            terminal = done

            for decision in range(first_decision_index, decision_limit):
                if terminal:
                    break
                matrix, scalar = state_to_tensors(state, device)
                mean, std = actor(matrix, scalar)
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample()
                action_norm_t = torch.tanh(raw_action)
                log_prob = (
                    dist.log_prob(raw_action) - torch.log(1 - action_norm_t.pow(2) + 1e-6)
                ).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                value = value_net(matrix, scalar).squeeze(-1)

                action_norm = action_norm_t.detach().cpu().numpy()[0]
                env_action = normalized_to_env_action(action_norm, args.action_scale)
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, env_action, args.mode, decision)
                terminal = done or decision == decision_limit - 1
                decision_metrics = collect_metrics(env, episode, args.items, args.month, args.mode, args.seed)
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        env_action,
                        decision_metrics,
                        "a2c",
                        args.items,
                        args.month,
                        args.mode,
                        args.seed,
                        decision_start_time=decision_start_time,
                    )
                )

                log_probs.append(log_prob.squeeze(0))
                entropies.append(entropy.squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(float(reward))
                dones.append(terminal)
                state = next_state
                if terminal:
                    break

            if rewards:
                with torch.no_grad():
                    if terminal:
                        next_value = torch.zeros(1, device=device)
                    else:
                        matrix, scalar = state_to_tensors(state, device)
                        next_value = value_net(matrix, scalar).squeeze(-1)

                values_t = torch.stack(values)
                log_probs_t = torch.stack(log_probs)
                entropies_t = torch.stack(entropies)
                advantages, returns = compute_gae(
                    rewards, dones, values_t, next_value, args.gamma, args.gae_lambda, device
                )

                policy_loss = -(log_probs_t * advantages.detach()).mean()
                value_loss = mse_loss(values_t, returns.detach())
                entropy_loss = -entropies_t.mean()
                loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(value_net.parameters()), args.max_grad_norm)
                optimizer.step()

            metrics = collect_metrics(env, episode, args.items, args.month, args.mode, args.seed)
            logger.write(metrics)
            viz.line([metrics["total_cost"]], [episode], win=win, update="append")
            print_episode("a2c", metrics)

            if metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                save_checkpoint(model_path, {"actor": actor, "value": value_net}, vars(args))
                write_best_config_csv(best_config_csv_path, episode_config_rows)
    finally:
        logger.close()


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
