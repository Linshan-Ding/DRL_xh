from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.training_utils import (
    BEST_CONFIG_HEADER,
    best_config_path,
    collect_resource_config,
    write_best_config_csv,
)


class BestConfigTest(unittest.TestCase):
    def test_collect_resource_config_counts_resources(self):
        robots = [
            SimpleNamespace(remove=False, rent="long"),
            SimpleNamespace(remove=False, rent="short"),
            SimpleNamespace(remove=True, rent="short"),
        ]
        pickers_area = {
            "area1": [
                SimpleNamespace(remove=False, rent="long"),
                SimpleNamespace(remove=False, rent="short"),
            ],
            "area2": [SimpleNamespace(remove=False, rent="short")],
            "area3": [SimpleNamespace(remove=True, rent="short")],
        }
        env = SimpleNamespace(
            area_ids=["area1", "area2", "area3"],
            robots=robots,
            pickers=[picker for pickers in pickers_area.values() for picker in pickers],
            pickers_area=pickers_area,
            adjust_robots=1,
            adjust_pickers_dict={"area1": 2, "area2": -1, "area3": 0},
            current_time=20,
        )
        metrics = {
            "total_cost": 10,
            "delay_cost": 1,
            "robot_cost": 2,
            "picker_cost": 7,
            "completed_orders": 3,
            "on_time_completed_orders": 2,
            "total_orders": 4,
            "completion_rate": 0.5,
            "average_picking_time": 12.5,
        }

        row = collect_resource_config(
            env,
            5,
            0,
            [1.4, 2, -1, 0],
            metrics,
            "ppo",
            2,
            1,
            "short",
            0,
            decision_start_time=0,
        )

        self.assertEqual(row["n_robots_total"], 2)
        self.assertEqual(row["n_robots_long"], 1)
        self.assertEqual(row["n_robots_short"], 1)
        self.assertEqual(row["n_pickers_total"], 3)
        self.assertEqual(row["n_pickers_area1"], 2)
        self.assertEqual(row["n_pickers_area2"], 1)
        self.assertEqual(row["n_pickers_area3"], 0)
        self.assertEqual(row["effective_picker_area2_delta"], -1)
        self.assertEqual(row["decision_end_time"], 20)

    def test_write_best_config_csv_overwrites(self):
        first_row = {key: "" for key in BEST_CONFIG_HEADER}
        first_row["algorithm"] = "ppo"
        first_row["total_cost"] = 100
        second_row = {key: "" for key in BEST_CONFIG_HEADER}
        second_row["algorithm"] = "a2c"
        second_row["total_cost"] = 50

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.csv"
            write_best_config_csv(path, [first_row, first_row])
            write_best_config_csv(path, [second_row])

            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "a2c")
        self.assertEqual(rows[0]["total_cost"], "50")

    def test_best_config_path_suffix(self):
        path = best_config_path("result/ppo", "short_i2_m01_seed0")
        self.assertEqual(path.name, "short_i2_m01_seed0_best_config.csv")

    def test_collect_resource_config_records_fixed_hybrid_initialization(self):
        robots = [
            SimpleNamespace(remove=False, rent="long"),
            SimpleNamespace(remove=False, rent="long"),
        ]
        pickers_area = {
            "area1": [SimpleNamespace(remove=False, rent="long")],
            "area2": [
                SimpleNamespace(remove=False, rent="long"),
                SimpleNamespace(remove=False, rent="long"),
            ],
            "area3": [
                SimpleNamespace(remove=False, rent="long"),
                SimpleNamespace(remove=False, rent="long"),
                SimpleNamespace(remove=False, rent="long"),
            ],
        }
        env = SimpleNamespace(
            area_ids=["area1", "area2", "area3"],
            robots=robots,
            pickers=[picker for pickers in pickers_area.values() for picker in pickers],
            pickers_area=pickers_area,
            adjust_robots=2,
            adjust_pickers_dict={"area1": 1, "area2": 2, "area3": 3},
            current_time=28800,
        )
        metrics = {
            "total_cost": 10,
            "delay_cost": 1,
            "robot_cost": 2,
            "picker_cost": 7,
            "completed_orders": 3,
            "on_time_completed_orders": 2,
            "total_orders": 4,
            "completion_rate": 0.5,
            "average_picking_time": 12.5,
        }

        row = collect_resource_config(
            env,
            5,
            0,
            [2, 1, 2, 3],
            metrics,
            "ppo",
            2,
            1,
            "fixed_hybrid",
            0,
            decision_start_time=0,
        )

        self.assertEqual(row["mode"], "fixed_hybrid")
        self.assertEqual(row["decision_index"], 0)
        self.assertEqual(row["n_robots_long"], 2)
        self.assertEqual(row["n_robots_short"], 0)
        self.assertEqual(row["n_pickers_long_area1"], 1)
        self.assertEqual(row["n_pickers_long_area2"], 2)
        self.assertEqual(row["n_pickers_long_area3"], 3)
        self.assertEqual(row["n_pickers_short_area1"], 0)
        self.assertEqual(row["n_pickers_short_area2"], 0)
        self.assertEqual(row["n_pickers_short_area3"], 0)


if __name__ == "__main__":
    unittest.main()
