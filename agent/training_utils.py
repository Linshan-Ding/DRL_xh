from __future__ import annotations

import copy
import csv
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None

    class _MissingTorchNN:
        Module = object

        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'torch'")

    nn = _MissingTorchNN()

from environment.class_public import Config, REPO_ROOT
from environment.warehouse_env import WarehouseEnv


DEFAULT_PARAMETERS = Config().parameters
FIXED_HYBRID_MODE = "fixed_hybrid"
MODES = ("short", "long", "hybrid", FIXED_HYBRID_MODE)
CSV_HEADER = [
    "episode",
    "total_cost",
    "delay_cost",
    "robot_cost",
    "picker_cost",
    "completed_orders",
    "on_time_completed_orders",
    "total_orders",
    "average_picking_time",
    "completion_rate",
    "item_count",
    "month",
    "mode",
    "seed",
]

BEST_CONFIG_HEADER = [
    "algorithm",
    "mode",
    "item_count",
    "month",
    "seed",
    "best_episode",
    "decision_index",
    "day_index",
    "decision_start_time",
    "decision_end_time",
    "action_robot_delta",
    "action_picker_area1_delta",
    "action_picker_area2_delta",
    "action_picker_area3_delta",
    "effective_robot_delta",
    "effective_picker_area1_delta",
    "effective_picker_area2_delta",
    "effective_picker_area3_delta",
    "n_robots_total",
    "n_robots_long",
    "n_robots_short",
    "n_pickers_total",
    "n_pickers_area1",
    "n_pickers_area2",
    "n_pickers_area3",
    "n_pickers_long_area1",
    "n_pickers_long_area2",
    "n_pickers_long_area3",
    "n_pickers_short_area1",
    "n_pickers_short_area2",
    "n_pickers_short_area3",
    "total_cost",
    "delay_cost",
    "robot_cost",
    "picker_cost",
    "completed_orders",
    "on_time_completed_orders",
    "total_orders",
    "completion_rate",
    "average_picking_time",
]


def repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_paths(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    resolved_dir = repo_path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir / f"{stem}.csv", resolved_dir / f"{stem}.pth"


def best_config_path(output_dir: str | Path, stem: str) -> Path:
    resolved_dir = repo_path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir / f"{stem}_best_config.csv"


def order_instance_path(parameters: dict[str, Any], item_count: int, month: int) -> Path:
    return (
        repo_path(parameters["paths"]["instance_dir"])
        / f"items_{item_count}"
        / f"orders_m{month:02d}.pkl"
    )


def case_stem(mode: str, item_count: int, month: int, seed: int) -> str:
    return f"{mode}_i{item_count}_m{month:02d}_seed{seed}"


def load_orders(parameters: dict[str, Any], item_count: int, month: int):
    order_path = order_instance_path(parameters, item_count, month)
    if not order_path.exists():
        raise FileNotFoundError(
            f"Order file not found: {order_path}. "
            "Generate configured instances first with: python -m data.generate_orders"
        )
    with order_path.open("rb") as f:
        return pickle.load(f)


def init_base_env(parameters: dict[str, Any]) -> WarehouseEnv:
    env = WarehouseEnv()
    experiment = parameters["experiment"]
    env.total_time = int(experiment["work_seconds_per_day"]) * int(experiment["total_days"])
    return env


def order_total_time(parameters: dict[str, Any], orders) -> int | None:
    if not orders:
        return None
    day_time = int(parameters["experiment"]["work_seconds_per_day"])
    last_arrival = max(order.arrive_time for order in orders)
    total_days = int(last_arrival // day_time) + 1
    return day_time * total_days


def max_decisions(mode: str, max_days: int) -> int:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == FIXED_HYBRID_MODE and int(max_days) < 2:
        raise ValueError("fixed_hybrid mode requires max_days >= 2")
    return 1 if mode == "long" else max_days


def fixed_hybrid_long_term_action(parameters: dict[str, Any], area_ids: list[str]) -> np.ndarray:
    fixed_config = parameters["experiment"][FIXED_HYBRID_MODE]
    robot_count = int(fixed_config["long_term_robots"])
    picker_counts = list(fixed_config["long_term_pickers_area"])
    if robot_count < 1:
        raise ValueError("experiment.fixed_hybrid.long_term_robots must be at least 1")
    if len(picker_counts) != len(area_ids):
        raise ValueError(
            "experiment.fixed_hybrid.long_term_pickers_area length must match "
            "the number of warehouse areas"
        )
    picker_counts = [int(count) for count in picker_counts]
    if any(count < 1 for count in picker_counts):
        raise ValueError(
            "experiment.fixed_hybrid.long_term_pickers_area values must be at least 1"
        )
    return np.asarray([robot_count, *picker_counts], dtype=np.float32)


def step_env(env: WarehouseEnv, action: np.ndarray, mode: str, decision_index: int):
    if mode == "long":
        return env.step(action, first_step=True, pattern="long")
    if mode == "hybrid":
        return env.step(action, first_step=(decision_index == 0))
    if mode == FIXED_HYBRID_MODE:
        if decision_index < 1:
            raise ValueError(
                "fixed_hybrid decision 0 is reserved for fixed long-term resources"
            )
        return env.step(action)
    return env.step(action)


def make_episode_env(base_env: WarehouseEnv, orders):
    env = copy.deepcopy(base_env)
    env.total_time = order_total_time(env.parameters, orders) or env.total_time
    state = env.reset(copy.deepcopy(orders))
    return env, state


def collect_metrics(
    env: WarehouseEnv,
    episode: int,
    item_count: int,
    month: int,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    delay_cost = sum(order.total_delay_cost(env.current_time) for order in env.orders_arrived)
    robot_cost = sum(robot.total_run_cost(env.current_time) for robot in env.robots_added)
    picker_cost = sum(picker.total_hire_cost(env.current_time) for picker in env.pickers_added)
    total_cost = delay_cost + robot_cost + picker_cost

    completed_orders = env.orders_completed
    completed_count = len(completed_orders)
    total_orders = len(env.orders_arrived)
    on_time = len([order for order in completed_orders if order.complete_time <= order.due_time])
    avg_picking_time = 0.0
    if completed_count > 0:
        avg_picking_time = sum(
            order.complete_time - order.arrive_time for order in completed_orders
        ) / completed_count
    completion_rate = on_time / total_orders if total_orders > 0 else 0.0

    return {
        "episode": episode,
        "total_cost": total_cost,
        "delay_cost": delay_cost,
        "robot_cost": robot_cost,
        "picker_cost": picker_cost,
        "completed_orders": completed_count,
        "on_time_completed_orders": on_time,
        "total_orders": total_orders,
        "average_picking_time": avg_picking_time,
        "completion_rate": completion_rate,
        "item_count": item_count,
        "month": month,
        "mode": mode,
        "seed": seed,
    }


def metrics_row(metrics: dict[str, Any]) -> list[Any]:
    return [metrics[key] for key in CSV_HEADER]


def _action_value(action: np.ndarray, index: int) -> float:
    action_array = np.asarray(action, dtype=float).reshape(-1)
    if index >= len(action_array):
        return 0.0
    return float(action_array[index])


def _area_values(values: dict[str, Any], area_ids: list[str]) -> list[Any]:
    return [values.get(area_id, 0) for area_id in area_ids[:3]]


def _pad_to_three(values: list[Any]) -> list[Any]:
    return values + [0] * (3 - len(values))


def collect_resource_config(
    env: WarehouseEnv,
    episode: int,
    decision: int,
    action: np.ndarray,
    metrics: dict[str, Any],
    algorithm: str,
    item_count: int,
    month: int,
    mode: str,
    seed: int,
    decision_start_time: float | None = None,
) -> dict[str, Any]:
    area_ids = list(env.area_ids)
    action_picker = [_action_value(action, index) for index in range(1, 4)]
    effective_picker = _pad_to_three(_area_values(env.adjust_pickers_dict or {}, area_ids))

    active_robots = [robot for robot in env.robots if not robot.remove]
    active_pickers = [picker for picker in env.pickers if not picker.remove]

    pickers_by_area = {
        area_id: [picker for picker in env.pickers_area[area_id] if not picker.remove]
        for area_id in area_ids
    }
    picker_total_by_area = _pad_to_three([len(pickers_by_area[area_id]) for area_id in area_ids[:3]])
    picker_long_by_area = _pad_to_three(
        [
            len([picker for picker in pickers_by_area[area_id] if picker.rent == "long"])
            for area_id in area_ids[:3]
        ]
    )
    picker_short_by_area = _pad_to_three(
        [
            len([picker for picker in pickers_by_area[area_id] if picker.rent == "short"])
            for area_id in area_ids[:3]
        ]
    )

    row = {
        "algorithm": algorithm,
        "mode": mode,
        "item_count": item_count,
        "month": month,
        "seed": seed,
        "best_episode": episode,
        "decision_index": decision,
        "day_index": decision + 1,
        "decision_start_time": decision_start_time if decision_start_time is not None else "",
        "decision_end_time": env.current_time,
        "action_robot_delta": _action_value(action, 0),
        "action_picker_area1_delta": action_picker[0],
        "action_picker_area2_delta": action_picker[1],
        "action_picker_area3_delta": action_picker[2],
        "effective_robot_delta": env.adjust_robots,
        "effective_picker_area1_delta": effective_picker[0],
        "effective_picker_area2_delta": effective_picker[1],
        "effective_picker_area3_delta": effective_picker[2],
        "n_robots_total": len(active_robots),
        "n_robots_long": len([robot for robot in active_robots if robot.rent == "long"]),
        "n_robots_short": len([robot for robot in active_robots if robot.rent == "short"]),
        "n_pickers_total": len(active_pickers),
        "n_pickers_area1": picker_total_by_area[0],
        "n_pickers_area2": picker_total_by_area[1],
        "n_pickers_area3": picker_total_by_area[2],
        "n_pickers_long_area1": picker_long_by_area[0],
        "n_pickers_long_area2": picker_long_by_area[1],
        "n_pickers_long_area3": picker_long_by_area[2],
        "n_pickers_short_area1": picker_short_by_area[0],
        "n_pickers_short_area2": picker_short_by_area[1],
        "n_pickers_short_area3": picker_short_by_area[2],
    }
    for key in (
        "total_cost",
        "delay_cost",
        "robot_cost",
        "picker_cost",
        "completed_orders",
        "on_time_completed_orders",
        "total_orders",
        "completion_rate",
        "average_picking_time",
    ):
        row[key] = metrics[key]
    return row


def initialize_episode_resources(
    env: WarehouseEnv,
    parameters: dict[str, Any],
    mode: str,
    episode: int,
    item_count: int,
    month: int,
    seed: int,
    algorithm: str,
) -> tuple[dict[str, Any], bool, int, list[dict[str, Any]]]:
    if mode != FIXED_HYBRID_MODE:
        return env.state, False, 0, []

    fixed_action = fixed_hybrid_long_term_action(parameters, list(env.area_ids))
    decision_start_time = env.current_time
    state, _reward, done = env.step(fixed_action, first_step=True)
    metrics = collect_metrics(env, episode, item_count, month, mode, seed)
    config_row = collect_resource_config(
        env,
        episode,
        0,
        fixed_action,
        metrics,
        algorithm,
        item_count,
        month,
        mode,
        seed,
        decision_start_time=decision_start_time,
    )
    return state, done, 1, [config_row]


def write_best_config_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BEST_CONFIG_HEADER)
        writer.writeheader()
        writer.writerows(rows)


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_HEADER)

    def write(self, metrics: dict[str, Any]) -> None:
        self.writer.writerow(metrics_row(metrics))
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class NullViz:
    def line(self, *args, **kwargs):
        return None


def make_visdom(enabled: bool, env_name: str, win: str, title: str | None = None):
    if not enabled:
        return NullViz(), win
    try:
        from visdom import Visdom

        viz = Visdom(env=env_name)
        viz.line(
            [0],
            [0],
            win=win,
            opts=dict(title=title or win, xlabel="Episode", ylabel="Total Cost"),
        )
        return viz, win
    except Exception as exc:
        print(f"Visdom disabled: {exc}")
        return NullViz(), win


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def save_checkpoint(path: Path, modules: dict[str, nn.Module], extra: dict | None = None) -> None:
    if torch is None:
        raise ModuleNotFoundError("No module named 'torch'")
    checkpoint = {name: module.state_dict() for name, module in modules.items()}
    if extra:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def print_episode(algorithm: str, metrics: dict[str, Any]) -> None:
    print(
        f"{algorithm.upper()} episode {metrics['episode']}: "
        f"cost={metrics['total_cost']:.2f}, "
        f"orders={metrics['completed_orders']}/{metrics['total_orders']}, "
        f"rate={metrics['completion_rate']:.4f}"
    )
