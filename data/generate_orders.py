from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from environment.class_public import Config, REPO_ROOT, print_config_source
from environment.class_warehouse import Order
from environment.warehouse_env import WarehouseEnv


ORDER_N_ITEMS_LIST = [2, 4, 6, 10]

MONTH_PARAMS = {
    1: [
        69.87, 75.26, 88.29, 83.17, 57.67, 191.07, 69.58, 206.19, 151.47, 127.39,
        131.82, 257.36, 125.44, 142.16, 102.46, 80.46, 88.79, 330.02, 254.30,
        117.99, 64.74, 125.44, 114.24, 71.64, 80.10, 110.13, 181.91, 56.55,
        121.47, 175.57, 100.58,
    ],
    2: [
        125.78, 146.12, 129.82, 103.58, 120.80, 120.83, 63.94, 63.23, 98.56,
        112.57, 304.18, 135.01, 64.90, 96.49, 87.84, 63.17, 62.83, 128.12,
        66.23, 79.54, 66.61, 56.67, 118.63, 66.23, 106.99, 279.11, 153.28,
        53.68, 89.71,
    ],
    3: [
        98.63, 81.29, 259.21, 127.52, 276.50, 189.26, 100.65, 82.79, 83.45,
        139.84, 73.37, 54.01, 79.11, 129.08, 92.46, 152.03, 71.87, 126.81,
        87.51, 182.32, 106.54, 100.60, 87.74, 140.75, 104.47, 62.08, 90.57,
        98.58, 85.60, 46.44, 86.55,
    ],
    4: [
        109.91, 108.85, 99.69, 107.91, 87.01, 83.12, 94.97, 214.69, 118.40,
        128.76, 114.06, 61.47, 132.52, 201.93, 63.60, 385.58, 72.08, 88.75,
        97.68, 200.32, 72.38, 93.35, 52.85, 84.67, 106.14, 150.11, 73.01,
        146.82, 102.91, 151.53,
    ],
    5: [
        120.83, 97.72, 58.62, 106.55, 145.36, 127.13, 42.44, 126.26, 81.32,
        130.03, 117.83, 95.13, 74.20, 92.04, 87.20, 297.07, 51.57, 120.28,
        292.37, 103.89, 93.45, 172.02, 61.27, 100.39, 85.12, 139.19, 213.61,
        129.63, 125.22, 93.35, 68.91,
    ],
    6: [
        80.58, 76.05, 99.40, 98.37, 101.32, 117.58, 195.32, 149.42, 104.80,
        53.75, 75.93, 83.39, 59.49, 121.30, 95.70, 202.61, 161.89, 42.10,
        356.55, 146.05, 123.93, 154.16, 387.37, 56.25, 65.41, 115.07, 173.48,
        156.56, 99.01, 131.64,
    ],
    7: [
        135.89, 54.01, 46.88, 278.14, 113.31, 113.89, 131.92, 96.25, 65.31,
        62.53, 55.64, 109.00, 75.67, 59.54, 65.13, 104.74, 68.95, 217.13,
        313.00, 115.35, 104.91, 62.17, 155.73, 97.55, 62.10, 88.98, 145.85,
        90.49, 82.05, 92.47, 89.10,
    ],
    8: [
        116.75, 72.83, 122.35, 103.94, 167.65, 176.96, 70.60, 209.91, 181.56,
        102.03, 270.30, 114.83, 116.43, 231.26, 104.19, 105.78, 119.87, 95.82,
        64.14, 158.36, 89.99, 71.78, 86.56, 152.57, 190.56, 157.47, 342.26,
        63.39, 101.06, 93.35,
    ],
    9: [
        126.38, 75.59, 83.75, 50.57, 113.71, 88.18, 64.00, 66.90, 117.09,
        98.49, 76.99, 60.62, 108.32, 62.32, 117.60, 157.03, 132.17, 59.96,
        143.89, 151.25, 77.19, 94.40, 71.46, 81.66, 118.28, 170.41, 390.10,
        103.16, 323.64, 82.82,
    ],
    10: [
        58.73, 76.19, 129.81, 107.98, 58.54, 104.20, 154.11, 67.89, 79.77,
        73.64, 110.42, 140.24, 74.22, 103.57, 66.22, 75.30, 90.19, 92.79,
        104.67, 66.13, 82.20, 61.47, 188.43, 128.08, 131.39, 90.98, 59.37,
        80.43, 124.08, 217.02, 85.94,
    ],
    11: [
        91.93, 162.33, 83.91, 90.13, 167.78, 89.30, 80.23, 192.77, 97.24,
        63.57, 79.13, 80.90, 73.19, 233.03, 195.28, 123.60, 153.50, 74.03,
        211.58, 63.14, 132.43, 95.64, 78.89, 106.04, 132.44, 69.11, 84.44,
        70.69, 163.57, 97.10,
    ],
    12: [
        58.09, 95.29, 72.75, 70.71, 71.23, 177.37, 294.96, 128.55, 92.15,
        108.71, 96.87, 61.27, 94.59, 75.40, 100.69, 67.64, 48.30, 93.07,
        118.67, 108.23, 74.32, 99.04, 116.53, 77.16, 126.68, 55.80, 193.81,
        98.52, 82.83, 87.01,
    ],
}


class GenerateData(Config):
    def __init__(
        self,
        warehouse: WarehouseEnv,
        total_seconds: int,
        poisson_parameters: Sequence[float] | None = None,
        order_n_items: int | None = None,
    ):
        super().__init__()
        self.parameter = self.parameters["order"]
        self.warehouse = warehouse
        self.total_seconds = total_seconds
        self.poisson_parameters = poisson_parameters
        self.order_n_items = order_n_items
        self.day_time = int(self.parameters["experiment"]["work_seconds_per_day"])

    def _arrival_interval(self, arrival_time: float) -> float:
        if self.poisson_parameters:
            day_count = int(arrival_time // self.day_time)
            index = min(day_count, len(self.poisson_parameters) - 1)
            return float(self.poisson_parameters[index])
        low, high = self.parameter["poisson_parameter"]
        return random.uniform(float(low), float(high))

    def generate_orders(self):
        orders = []
        order_id = 0
        arrival_time = 0.0
        warehouse_items = list(self.warehouse.items.values())

        while arrival_time < self.total_seconds:
            n_items = self.order_n_items
            if n_items is None:
                low, high = self.parameter["order_n_items"]
                n_items = random.randint(int(low), int(high))
            n_items = random.choice([n_items - 1, n_items, n_items + 1])
            n_items = max(1, min(n_items, len(warehouse_items)))

            arrival_time += random.expovariate(1 / self._arrival_interval(arrival_time))
            if arrival_time >= self.total_seconds:
                break

            items = copy.deepcopy(random.sample(warehouse_items, n_items))
            due_time = arrival_time + random.choice(self.parameter["due_time_list"])
            order_id += 1
            orders.append(Order(order_id, items, arrival_time, due_time))

        return orders


def repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def validate_choices(values: Iterable[int], allowed: Iterable[int], label: str) -> list[int]:
    allowed_values = set(allowed)
    selected = list(values)
    invalid = [value for value in selected if value not in allowed_values]
    if invalid:
        allowed_text = ", ".join(str(value) for value in sorted(allowed_values))
        invalid_text = ", ".join(str(value) for value in invalid)
        raise ValueError(f"Unsupported {label}: {invalid_text}. Valid values: {allowed_text}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate monthly order instances for DRL_Warehouse.")
    return parser


def save_orders(output_dir: Path, item_count: int, month: int, orders) -> Path:
    item_dir = output_dir / f"items_{item_count}"
    item_dir.mkdir(parents=True, exist_ok=True)
    output_path = item_dir / f"orders_m{month:02d}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(orders, f)
    print(f"Saved {len(orders)} orders to {output_path}")
    return output_path


def generate_cases(config: dict, warehouse: WarehouseEnv, output_dir: Path) -> None:
    experiment = config["experiment"]
    item_counts = validate_choices(experiment["item_scenarios"], ORDER_N_ITEMS_LIST, "item count")
    months = validate_choices(experiment["months"], MONTH_PARAMS, "month")
    work_seconds_per_day = int(config["experiment"]["work_seconds_per_day"])
    seed = int(experiment["seed"])

    for item_count in item_counts:
        for month in months:
            random.seed(seed + item_count * 100 + month - 1)
            np.random.seed(seed + item_count * 100 + month - 1)

            poisson_parameters = MONTH_PARAMS[month]
            total_seconds = work_seconds_per_day * len(poisson_parameters)
            generator = GenerateData(
                warehouse,
                total_seconds,
                poisson_parameters=poisson_parameters,
                order_n_items=item_count,
            )
            orders = generator.generate_orders()
            save_orders(output_dir, item_count, month, orders)


def main() -> None:
    build_parser().parse_args()
    print_config_source()
    config = Config().parameters
    output_dir = repo_path(config["paths"]["instance_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = WarehouseEnv()
    generate_cases(config, warehouse, output_dir)


if __name__ == "__main__":
    main()
