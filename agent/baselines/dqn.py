from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.baselines.common import (
    BranchingQNetwork,
    CsvLogger,
    DiscreteReplayBuffer,
    RunningRewardNormalizer,
    add_common_args,
    best_config_path,
    collect_metrics,
    collect_resource_config,
    discrete_action_cardinalities,
    discrete_indices_to_env_action,
    hard_update,
    init_base_env,
    initialize_episode_resources,
    is_result_candidate,
    linear_exploration_scale,
    load_orders,
    make_episode_env,
    make_visdom,
    max_decisions,
    max_discrete_action_cardinalities,
    output_path,
    print_config_source,
    print_episode,
    set_seed,
    state_to_arrays,
    state_to_tensors,
    step_env,
    use_deterministic_action,
    use_random_action,
    write_best_config_csv,
)


def build_dqn_parser(algorithm: str = "dqn") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{algorithm.upper()} baseline for warehouse resource allocation."
    )
    add_common_args(parser)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument(
        "--learning-starts",
        "--warmup-steps",
        dest="learning_starts",
        type=int,
        default=64,
    )
    parser.add_argument("--initial-epsilon", type=float, default=1.0)
    parser.add_argument("--final-epsilon", type=float, default=0.05)
    parser.add_argument("--exploration-decay-fraction", type=float, default=0.8)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return build_dqn_parser("dqn")


def greedy_branch_actions(
    branch_q_values: Sequence[torch.Tensor],
    valid_cardinalities: torch.Tensor,
) -> torch.Tensor:
    """Choose the best valid action independently in every discrete branch."""
    actions = []
    for branch_index, q_values in enumerate(branch_q_values):
        counts = valid_cardinalities[:, branch_index]
        positions = torch.arange(q_values.shape[1], device=q_values.device).unsqueeze(0)
        masked_q = q_values.masked_fill(positions >= counts.unsqueeze(1), -torch.inf)
        actions.append(masked_q.argmax(dim=1))
    return torch.stack(actions, dim=1)


def gather_branch_q(
    branch_q_values: Sequence[torch.Tensor],
    action_indices: torch.Tensor,
) -> torch.Tensor:
    selected = [
        q_values.gather(1, action_indices[:, index : index + 1])
        for index, q_values in enumerate(branch_q_values)
    ]
    return torch.cat(selected, dim=1).mean(dim=1, keepdim=True)


def select_target_actions(
    online_branch_q: Sequence[torch.Tensor],
    target_branch_q: Sequence[torch.Tensor],
    valid_cardinalities: torch.Tensor,
    double_dqn: bool,
) -> torch.Tensor:
    selection_q = online_branch_q if double_dqn else target_branch_q
    return greedy_branch_actions(selection_q, valid_cardinalities)


def update_dqn(
    online_network,
    target_network,
    optimizer,
    buffer,
    reward_normalizer,
    args,
    device,
    double_dqn: bool = False,
) -> float:
    (
        matrix,
        scalar,
        action,
        reward,
        next_matrix,
        next_scalar,
        next_cardinalities,
        done,
    ) = buffer.sample(args.batch_size, device, reward_normalizer)

    current_q = gather_branch_q(online_network(matrix, scalar), action)
    with torch.no_grad():
        target_next_branches = target_network(next_matrix, next_scalar)
        online_next_branches = (
            online_network(next_matrix, next_scalar)
            if double_dqn
            else target_next_branches
        )
        next_action = select_target_actions(
            online_next_branches,
            target_next_branches,
            next_cardinalities,
            double_dqn,
        )
        next_q = gather_branch_q(target_next_branches, next_action)
        target_q = reward + args.gamma * (1 - done) * next_q

    loss = nn.functional.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), args.max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def select_greedy_action(
    network: BranchingQNetwork,
    state: dict,
    valid_cardinalities: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    matrix, scalar = state_to_tensors(state, device)
    cardinalities = torch.as_tensor(
        valid_cardinalities,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        action = greedy_branch_actions(network(matrix, scalar), cardinalities)
    return action.cpu().numpy()[0]


def random_discrete_action(valid_cardinalities: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.random.randint(int(count)) for count in valid_cardinalities],
        dtype=np.int64,
    )


def train_dqn(args, algorithm: str, double_dqn: bool) -> None:
    print_config_source()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.target_update_interval < 1:
        raise ValueError("target_update_interval must be positive")
    if not 0 <= args.initial_epsilon <= 1 or not 0 <= args.final_epsilon <= 1:
        raise ValueError("epsilon values must be in [0, 1]")

    orders = load_orders(args.items, args.month)
    base_env = init_base_env()
    decision_limit = max_decisions(args.mode, args.max_days)
    action_dim = base_env.N_a + 1
    scalar_dim = base_env.N_a + 1
    matrix_shape = (3, base_env.N_w, base_env.N_l)
    branch_sizes = max_discrete_action_cardinalities(
        args.mode,
        action_dim,
        args.action_scale,
        args.long_robot_max,
        args.long_picker_max,
    )

    online_network = BranchingQNetwork(
        base_env.N_w,
        base_env.N_l,
        scalar_dim,
        branch_sizes,
    ).to(device)
    target_network = BranchingQNetwork(
        base_env.N_w,
        base_env.N_l,
        scalar_dim,
        branch_sizes,
    ).to(device)
    hard_update(online_network, target_network)
    optimizer = optim.Adam(online_network.parameters(), lr=args.learning_rate)
    buffer = DiscreteReplayBuffer(args.buffer_size, matrix_shape, scalar_dim, action_dim)
    reward_normalizer = RunningRewardNormalizer()

    csv_path = output_path(args, algorithm)
    best_config_csv_path = best_config_path(args, algorithm)
    logger = CsvLogger(csv_path)
    viz, win = make_visdom(args, algorithm)
    best_cost = math.inf
    global_step = 0
    update_step = 0
    total_steps = args.episodes * decision_limit
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
                algorithm,
            )
            deterministic = use_deterministic_action(
                episode,
                args.episodes,
                args.deterministic_tail_fraction,
            )

            for decision in range(first_decision_index, decision_limit):
                if done:
                    break
                matrix_np, scalar_np = state_to_arrays(state)
                valid_cardinalities = discrete_action_cardinalities(
                    args.mode,
                    decision,
                    action_dim,
                    args.action_scale,
                    args.long_robot_max,
                    args.long_picker_max,
                )

                if deterministic:
                    action_indices = select_greedy_action(
                        online_network,
                        state,
                        valid_cardinalities,
                        device,
                    )
                else:
                    epsilon = linear_exploration_scale(
                        global_step,
                        total_steps,
                        args.initial_epsilon,
                        args.final_epsilon,
                        args.exploration_decay_fraction,
                    )
                    if use_random_action(global_step, args.learning_starts) or np.random.random() < epsilon:
                        action_indices = random_discrete_action(valid_cardinalities)
                    else:
                        action_indices = select_greedy_action(
                            online_network,
                            state,
                            valid_cardinalities,
                            device,
                        )

                env_action = discrete_indices_to_env_action(
                    action_indices,
                    args.mode,
                    decision,
                    args.action_scale,
                    args.long_robot_max,
                    args.long_picker_max,
                )
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, env_action, args.mode, decision)
                terminal = done or decision == decision_limit - 1
                decision_metrics = collect_metrics(
                    env,
                    episode,
                    args.items,
                    args.month,
                    args.mode,
                    args.seed,
                )
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        env_action,
                        decision_metrics,
                        algorithm,
                        args.items,
                        args.month,
                        args.mode,
                        args.seed,
                        decision_start_time=decision_start_time,
                    )
                )

                next_matrix_np, next_scalar_np = state_to_arrays(next_state)
                if not deterministic:
                    next_cardinalities = discrete_action_cardinalities(
                        args.mode,
                        decision + 1,
                        action_dim,
                        args.action_scale,
                        args.long_robot_max,
                        args.long_picker_max,
                    )
                    reward_normalizer.update(float(reward))
                    buffer.add(
                        matrix_np,
                        scalar_np,
                        action_indices,
                        float(reward),
                        next_matrix_np,
                        next_scalar_np,
                        next_cardinalities,
                        terminal,
                    )
                    if buffer.size >= args.batch_size and global_step + 1 >= args.learning_starts:
                        update_dqn(
                            online_network,
                            target_network,
                            optimizer,
                            buffer,
                            reward_normalizer,
                            args,
                            device,
                            double_dqn=double_dqn,
                        )
                        update_step += 1
                        if update_step % args.target_update_interval == 0:
                            hard_update(online_network, target_network)

                state = next_state
                global_step += 1
                if terminal:
                    break

            metrics = collect_metrics(
                env,
                episode,
                args.items,
                args.month,
                args.mode,
                args.seed,
            )
            logger.write(metrics)
            viz.line([metrics["total_cost"]], [episode], win=win, update="append")
            print_episode(algorithm, metrics)

            best_candidate = is_result_candidate(deterministic, args.deterministic_tail_fraction)
            if best_candidate and metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                write_best_config_csv(best_config_csv_path, episode_config_rows)
    finally:
        logger.close()


def train(args) -> None:
    train_dqn(args, algorithm="dqn", double_dqn=False)


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
