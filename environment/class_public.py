from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.json"

REQUIRED_SECTIONS = (
    "warehouse",
    "robot",
    "picker",
    "order",
    "item",
    "ppo",
    "experiment",
    "paths",
)
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def resolve_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def print_config_source() -> None:
    print(f"Using config: {DEFAULT_CONFIG_PATH}")


def _as_tuple(parameters: dict[str, Any], section: str, key: str) -> None:
    value = parameters[section][key]
    if isinstance(value, list):
        parameters[section][key] = tuple(value)


def _normalize(parameters: dict[str, Any]) -> dict[str, Any]:
    _as_tuple(parameters, "warehouse", "depot_position")
    _as_tuple(parameters, "order", "poisson_parameter")
    _as_tuple(parameters, "order", "order_n_arrival")
    _as_tuple(parameters, "order", "order_n_items")
    return parameters


def validate_config(parameters: Mapping[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in parameters]
    if missing:
        raise KeyError(f"Missing config sections: {', '.join(missing)}")

    warehouse = parameters["warehouse"]
    for key in ("shelf_capacity", "shelf_levels", "area_num", "aisle_num"):
        if int(warehouse[key]) <= 0:
            raise ValueError(f"warehouse.{key} must be positive")

    experiment = parameters["experiment"]
    mode = experiment["mode"]
    valid_modes = ("short", "long", "hybrid", "fixed_hybrid")
    if mode is not None and mode not in valid_modes:
        raise ValueError(f"experiment.mode must be one of: {', '.join(valid_modes)}")
    if int(experiment["episodes"]) <= 0:
        raise ValueError("experiment.episodes must be positive")
    if int(experiment["max_days"]) <= 0:
        raise ValueError("experiment.max_days must be positive")
    if mode == "fixed_hybrid" and int(experiment["max_days"]) < 2:
        raise ValueError("fixed_hybrid mode requires experiment.max_days >= 2")

    fixed_hybrid = experiment["fixed_hybrid"]

    if int(fixed_hybrid["long_term_robots"]) < 1:
        raise ValueError("experiment.fixed_hybrid.long_term_robots must be at least 1")

    picker_counts = fixed_hybrid["long_term_pickers_area"]
    if not isinstance(picker_counts, list):
        raise ValueError("experiment.fixed_hybrid.long_term_pickers_area must be a list")
    if len(picker_counts) != int(warehouse["area_num"]):
        raise ValueError(
            "experiment.fixed_hybrid.long_term_pickers_area length must equal "
            "warehouse.area_num"
        )
    if any(int(count) < 1 for count in picker_counts):
        raise ValueError(
            "experiment.fixed_hybrid.long_term_pickers_area values must be at least 1"
        )


def load_config() -> dict[str, Any]:
    path = resolve_config_path()
    cache_key = str(path.resolve())
    if cache_key in _CONFIG_CACHE:
        return copy.deepcopy(_CONFIG_CACHE[cache_key])

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        parameters = json.load(f)

    parameters = _normalize(parameters)
    validate_config(parameters)
    _CONFIG_CACHE[cache_key] = copy.deepcopy(parameters)
    return parameters


class Config:
    def __init__(self):
        self.config_path = resolve_config_path()
        self.parameters = load_config()

    def parameter(self) -> dict[str, Any]:
        return copy.deepcopy(self.parameters)
