from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.baselines.common import (
    CsvLogger,
    QNetwork,
    ReplayBuffer,
    RunningRewardNormalizer,
    SquashedGaussianActor,
    add_common_args,
    best_config_path,
    collect_metrics,
    collect_resource_config,
    hard_update,
    init_base_env,
    initialize_episode_resources,
    is_result_candidate,
    load_orders,
    make_episode_env,
    make_visdom,
    max_decisions,
    normalized_to_env_action,
    output_path,
    print_episode,
    print_config_source,
    set_seed,
    soft_update,
    state_to_arrays,
    state_to_tensors,
    step_env,
    use_deterministic_action,
    use_random_action,
    write_best_config_csv,
)


def build_parser():
    parser = argparse.ArgumentParser(description="SAC baseline for warehouse resource allocation.")
    add_common_args(parser)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument(
        "--learning-starts",
        "--warmup-steps",
        dest="learning_starts",
        type=int,
        default=64,
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def update_sac(
    actor,
    critic1,
    critic2,
    critic1_target,
    critic2_target,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    log_alpha,
    alpha_optimizer,
    target_entropy,
    buffer,
    reward_normalizer,
    args,
    device,
):
    matrix, scalar, action, reward, next_matrix, next_scalar, done = buffer.sample(
        args.batch_size,
        device,
        reward_normalizer,
    )
    alpha = log_alpha.exp()

    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_matrix, next_scalar)
        target_q1 = critic1_target(next_matrix, next_scalar, next_action)
        target_q2 = critic2_target(next_matrix, next_scalar, next_action)
        target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_prob
        target = reward + args.gamma * (1 - done) * target_q

    critic1_loss = nn.functional.mse_loss(critic1(matrix, scalar, action), target)
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    nn.utils.clip_grad_norm_(critic1.parameters(), args.max_grad_norm)
    critic1_optimizer.step()

    critic2_loss = nn.functional.mse_loss(critic2(matrix, scalar, action), target)
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    nn.utils.clip_grad_norm_(critic2.parameters(), args.max_grad_norm)
    critic2_optimizer.step()

    new_action, log_prob = actor.sample(matrix, scalar)
    q1_new = critic1(matrix, scalar, new_action)
    q2_new = critic2(matrix, scalar, new_action)
    actor_loss = (alpha.detach() * log_prob - torch.minimum(q1_new, q2_new)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_update(critic1, critic1_target, args.tau)
    soft_update(critic2, critic2_target, args.tau)


def train(args):
    print_config_source()
    set_seed(args.seed)
    device = torch.device(args.device)

    orders = load_orders(args.items, args.month)
    base_env = init_base_env()
    decision_limit = max_decisions(args.mode, args.max_days)
    action_dim = base_env.N_a + 1
    scalar_dim = base_env.N_a + 1
    matrix_shape = (3, base_env.N_w, base_env.N_l)

    actor = SquashedGaussianActor(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic1 = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic2 = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic1_target = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic2_target = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    hard_update(critic1, critic1_target)
    hard_update(critic2, critic2_target)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=args.critic_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -float(action_dim)
    buffer = ReplayBuffer(args.buffer_size, matrix_shape, scalar_dim, action_dim)
    reward_normalizer = RunningRewardNormalizer()

    csv_path = output_path(args, "sac")
    best_config_csv_path = best_config_path(args, "sac")
    logger = CsvLogger(csv_path)
    viz, win = make_visdom(args, "sac")
    best_cost = math.inf
    global_step = 0
    write_best_config_csv(best_config_csv_path, [])

    try:
        for episode in range(1, args.episodes + 1):
            env, state = make_episode_env(base_env, orders)
            state, done, first_decision_index, episode_config_rows = initialize_episode_resources(
                env,
                args.mode,
                episode,
                args.items,
                args.month,
                args.seed,
                "sac",
            )
            for decision in range(first_decision_index, decision_limit):
                if done:
                    break
                matrix_np, scalar_np = state_to_arrays(state)
                deterministic = use_deterministic_action(
                    episode,
                    args.episodes,
                    args.deterministic_tail_fraction,
                )
                if use_random_action(global_step, args.learning_starts):
                    action_norm = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
                else:
                    matrix, scalar = state_to_tensors(state, device)
                    if deterministic:
                        action_norm = actor.deterministic(matrix, scalar)
                    else:
                        action_norm, _ = actor.sample(matrix, scalar)
                    action_norm = action_norm.detach().cpu().numpy()[0]

                env_action = normalized_to_env_action(
                    action_norm,
                    args.action_scale,
                    args.mode,
                    decision,
                    args.long_robot_center,
                    args.long_picker_center,
                )
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
                        "sac",
                        args.items,
                        args.month,
                        args.mode,
                        args.seed,
                        decision_start_time=decision_start_time,
                    )
                )
                next_matrix_np, next_scalar_np = state_to_arrays(next_state)
                if not deterministic:
                    reward_normalizer.update(float(reward))
                    buffer.add(
                        matrix_np,
                        scalar_np,
                        action_norm,
                        float(reward),
                        next_matrix_np,
                        next_scalar_np,
                        terminal,
                    )

                    if (
                        buffer.size >= args.batch_size
                        and global_step + 1 >= args.learning_starts
                    ):
                        update_sac(
                            actor,
                            critic1,
                            critic2,
                            critic1_target,
                            critic2_target,
                            actor_optimizer,
                            critic1_optimizer,
                            critic2_optimizer,
                            log_alpha,
                            alpha_optimizer,
                            target_entropy,
                            buffer,
                            reward_normalizer,
                            args,
                            device,
                        )

                state = next_state
                global_step += 1
                if terminal:
                    break

            metrics = collect_metrics(env, episode, args.items, args.month, args.mode, args.seed)
            logger.write(metrics)
            viz.line([metrics["total_cost"]], [episode], win=win, update="append")
            print_episode("sac", metrics)

            best_candidate = is_result_candidate(deterministic, args.deterministic_tail_fraction)
            if best_candidate and metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                write_best_config_csv(best_config_csv_path, episode_config_rows)
    finally:
        logger.close()


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
