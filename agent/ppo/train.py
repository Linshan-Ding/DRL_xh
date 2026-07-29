from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from agent.training_utils import (
    CSV_HEADER,
    best_config_path,
    case_stem,
    collect_metrics,
    collect_resource_config,
    init_base_env,
    initialize_episode_resources,
    load_orders,
    make_episode_env,
    make_visdom,
    max_decisions,
    order_instance_path,
    output_paths,
    set_seed,
    step_env,
    write_best_config_csv,
)
from environment.class_public import load_config, print_config_source
from environment.warehouse_env import WarehouseEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unified PPO agents for DRL_Warehouse.")
    return parser


def make_networks(env: WarehouseEnv, parameters: dict[str, Any]):
    from agent.ppo.networks import PolicyNetwork, ValueNetwork

    ppo = parameters["ppo"]
    scalar_dim = env.N_a + 1
    action_dim = env.N_a + 1
    policy = PolicyNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        output_dim=action_dim,
        hidden_dim=ppo["hidden_dim"],
        feature_dim=ppo["feature_dim"],
        attention_heads=ppo["attention_heads"],
        initial_log_std=ppo["initial_log_std"],
        min_log_std=ppo["min_log_std"],
        max_log_std=ppo["max_log_std"],
    )
    value = ValueNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        hidden_dim=ppo["hidden_dim"],
        feature_dim=ppo["feature_dim"],
    )
    return policy, value


def print_training_parameters(
    parameters: dict[str, Any],
    env: WarehouseEnv,
    decision_limit: int,
    order_path: Path,
    csv_path: Path,
    model_path: Path,
    best_config_csv_path: Path,
) -> None:
    experiment = parameters["experiment"]
    ppo = parameters["ppo"]
    warehouse = parameters["warehouse"]

    lines = [
        "Training parameters:",
        f"  experiment.mode: {experiment['mode']}",
        f"  experiment.item_scenario: {experiment['item_scenario']}",
        f"  experiment.month: {experiment['month']}",
        f"  experiment.episodes: {experiment['episodes']}",
        f"  experiment.seed: {experiment['seed']}",
        f"  experiment.max_days: {experiment['max_days']}",
        f"  experiment.decision_limit: {decision_limit}",
        f"  experiment.total_days: {experiment['total_days']}",
        f"  experiment.work_seconds_per_day: {experiment['work_seconds_per_day']}",
        f"  experiment.device: {experiment['device']}",
        f"  experiment.visdom: {experiment['visdom']}",
    ]
    if experiment["mode"] == "fixed_hybrid":
        fixed_hybrid = experiment["fixed_hybrid"]
        lines.extend(
            [
                f"  fixed_hybrid.long_term_robots: {fixed_hybrid['long_term_robots']}",
                f"  fixed_hybrid.long_term_pickers_area: {fixed_hybrid['long_term_pickers_area']}",
            ]
        )

    lines.extend(
        [
            f"  paths.orders: {order_path}",
            f"  paths.ppo_csv: {csv_path}",
            f"  paths.ppo_model: {model_path}",
            f"  paths.best_config_csv: {best_config_csv_path}",
            f"  ppo.learning_rate: {ppo['learning_rate']}",
            f"  ppo.gamma: {ppo['gamma']}",
            f"  ppo.batch_size: {ppo['batch_size']}",
            f"  ppo.n_epochs: {ppo['n_epochs']}",
            f"  ppo.gae_lambda: {ppo['gae_lambda']}",
            f"  ppo.clip_range: {ppo['clip_range']}",
            f"  ppo.initial_entropy_coeff: {ppo['initial_entropy_coeff']}",
            f"  ppo.min_entropy_coeff: {ppo['min_entropy_coeff']}",
            f"  ppo.entropy_coeff_decay: {ppo['entropy_coeff_decay']}",
            f"  ppo.hidden_dim: {ppo['hidden_dim']}",
            f"  ppo.feature_dim: {ppo['feature_dim']}",
            f"  ppo.attention_heads: {ppo['attention_heads']}",
            f"  ppo.initial_log_std: {ppo['initial_log_std']}",
            f"  ppo.min_log_std: {ppo['min_log_std']}",
            f"  ppo.max_log_std: {ppo['max_log_std']}",
            f"  warehouse.area_num: {warehouse['area_num']}",
            f"  warehouse.aisle_num: {warehouse['aisle_num']}",
            f"  warehouse.shelf_capacity: {warehouse['shelf_capacity']}",
            f"  warehouse.shelf_levels: {warehouse['shelf_levels']}",
            f"  env.total_time: {env.total_time}",
        ]
    )
    print("\n".join(lines))


def _done_for_update(mode: str, done: bool) -> bool:
    return True if mode == "long" else done


def _should_update(
    mode: str,
    buffer_len: int,
    batch_size: int,
    episode: int,
    total_episodes: int,
) -> bool:
    if mode != "long":
        return True
    return buffer_len >= batch_size


def _format_update_diagnostics(episode: int, stats: dict[str, float]) -> str:
    return (
        f"PPO update episode {episode}: "
        f"mean_reward={stats['mean_reward']:.4f}, "
        f"std_reward={stats['std_reward']:.4f}, "
        f"mean_return={stats['mean_return']:.4f}, "
        f"policy_loss={stats['policy_loss']:.4f}, "
        f"value_loss={stats['value_loss']:.4f}, "
        f"entropy={stats['entropy']:.4f}, "
        f"mean_action_std={stats['mean_action_std']:.4f}"
    )


def train(parameters: dict[str, Any]) -> Path:
    import torch

    from agent.ppo.agent import PPOAgent

    experiment = parameters["experiment"]
    mode = experiment["mode"]
    item_count = int(experiment["item_scenario"])
    month = int(experiment["month"])
    episodes = int(experiment["episodes"])
    seed = int(experiment["seed"])
    decision_limit = max_decisions(mode, int(experiment["max_days"]))
    set_seed(seed)

    base_env = init_base_env(parameters)
    policy, value = make_networks(base_env, parameters)
    agent = PPOAgent(policy, value, parameters=parameters, device=experiment["device"])
    stem = case_stem(mode, item_count, month, seed)
    csv_path, model_path = output_paths(
        parameters["paths"]["ppo_output_dir"],
        stem,
    )
    best_config_csv_path = best_config_path(parameters["paths"]["ppo_output_dir"], stem)
    order_path = order_instance_path(parameters, item_count, month)
    print_training_parameters(
        parameters,
        base_env,
        decision_limit,
        order_path,
        csv_path,
        model_path,
        best_config_csv_path,
    )
    orders = load_orders(parameters, item_count, month)
    viz, viz_win = make_visdom(
        bool(experiment["visdom"]),
        "DRL_PPO",
        f"ppo_{stem}",
    )

    best_cost = float("inf")
    if mode == "long":
        agent.buffer.clear()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for episode in range(1, episodes + 1):
            env, state = make_episode_env(base_env, orders)
            if mode != "long":
                agent.buffer.clear()
            state, done, first_decision_index, episode_config_rows = initialize_episode_resources(
                env,
                parameters,
                mode,
                episode,
                item_count,
                month,
                seed,
                "ppo",
            )
            learned_steps = 0

            for decision in range(first_decision_index, decision_limit):
                if done:
                    break
                action, log_prob, value_estimate, matrix_state, scalar_state = agent.select_action(state)
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, action, mode, decision)
                learned_steps += 1
                agent.buffer.add(
                    matrix_state,
                    scalar_state,
                    action,
                    log_prob,
                    reward,
                    _done_for_update(mode, done),
                    value_estimate,
                    maxlen=agent.batch_size if mode == "long" else None,
                )
                decision_metrics = collect_metrics(env, episode, item_count, month, mode, seed)
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        action,
                        decision_metrics,
                        "ppo",
                        item_count,
                        month,
                        mode,
                        seed,
                        decision_start_time=decision_start_time,
                    )
                )
                state = next_state

            updated = False
            if learned_steps > 0 and _should_update(mode, len(agent.buffer), agent.batch_size, episode, episodes):
                updated = agent.update(clear_buffer=(mode != "long"))
            metrics = collect_metrics(env, episode, item_count, month, mode, seed)
            writer.writerow([metrics[key] for key in CSV_HEADER])
            f.flush()
            viz.line([metrics["total_cost"]], [episode], win=viz_win, update="append")
            if updated and agent.last_update_stats:
                print(_format_update_diagnostics(episode, agent.last_update_stats))

            if metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                torch.save(
                    {
                        "policy": agent.policy.state_dict(),
                        "value": agent.value_network.state_dict(),
                        "config": parameters,
                        "metrics": metrics,
                    },
                    model_path,
                )
                write_best_config_csv(best_config_csv_path, episode_config_rows)

            print(
                f"PPO {mode} episode {episode}: "
                f"cost={metrics['total_cost']:.2f}, "
                f"orders={metrics['completed_orders']}/{metrics['total_orders']}, "
                f"rate={metrics['completion_rate']:.4f}"
            )

    return csv_path


def main() -> None:
    build_parser().parse_args()
    print_config_source()
    parameters = load_config()
    csv_path = train(parameters)
    print(f"Saved PPO metrics to {csv_path}")


if __name__ == "__main__":
    main()
