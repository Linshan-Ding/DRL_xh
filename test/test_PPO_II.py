from __future__ import annotations

import copy
import pickle
import unittest

from environment.class_public import Config, REPO_ROOT
from environment.warehouse_env import WarehouseEnv


class EnvironmentSmokeTest(unittest.TestCase):
    def test_items_2_month_1_reset_and_step(self):
        parameters = Config().parameters
        order_path = REPO_ROOT / parameters["paths"]["instance_dir"] / "items_2" / "orders_m01.pkl"
        self.assertTrue(order_path.exists(), f"Missing fixture: {order_path}")

        with order_path.open("rb") as f:
            orders = pickle.load(f)

        env = WarehouseEnv()
        env.total_time = (
            parameters["experiment"]["work_seconds_per_day"]
            * parameters["experiment"]["total_days"]
        )
        state = env.reset(orders)
        self.assertIn("robot_queue_list", state)

        action = [1, 1, 1, 1]
        next_state, reward, done = env.step(action, first_step=True)
        self.assertIn("n_robots", next_state)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(done, bool)

    def test_fixed_hybrid_initializes_long_resources_then_adjusts_short_resources(self):
        from agent.training_utils import initialize_episode_resources, step_env

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["fixed_hybrid"] = {
            "long_term_robots": 2,
            "long_term_pickers_area": [1, 2, 3],
        }
        env = WarehouseEnv()
        env.total_time = parameters["experiment"]["work_seconds_per_day"] * 2
        env.reset([])

        state, done, first_decision_index, rows = initialize_episode_resources(
            env,
            parameters,
            "fixed_hybrid",
            episode=1,
            item_count=2,
            month=1,
            seed=0,
            algorithm="test",
        )

        self.assertFalse(done)
        self.assertEqual(first_decision_index, 1)
        self.assertIn("n_robots", state)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "fixed_hybrid")
        self.assertEqual(rows[0]["decision_index"], 0)
        self.assertEqual(rows[0]["n_robots_long"], 2)
        self.assertEqual(rows[0]["n_pickers_long_area1"], 1)
        self.assertEqual(rows[0]["n_pickers_long_area2"], 2)
        self.assertEqual(rows[0]["n_pickers_long_area3"], 3)

        step_env(env, [2, 1, 1, 1], "fixed_hybrid", first_decision_index)
        self.assertEqual(len([robot for robot in env.robots if robot.rent == "long"]), 2)
        self.assertEqual(len([robot for robot in env.robots if robot.rent == "short"]), 2)
        self.assertEqual(
            [len([picker for picker in env.pickers_area[area_id] if picker.rent == "long"])
             for area_id in env.area_ids],
            [1, 2, 3],
        )
        self.assertEqual(
            [len([picker for picker in env.pickers_area[area_id] if picker.rent == "short"])
             for area_id in env.area_ids],
            [1, 1, 1],
        )

        step_env(env, [-2, -1, -1, -1], "fixed_hybrid", first_decision_index + 1)
        self.assertEqual(len([robot for robot in env.robots if robot.rent == "long"]), 2)
        self.assertEqual(len([robot for robot in env.robots if robot.rent == "short"]), 0)
        self.assertEqual(
            [len([picker for picker in env.pickers_area[area_id] if picker.rent == "long"])
             for area_id in env.area_ids],
            [1, 2, 3],
        )
        self.assertEqual(
            [len([picker for picker in env.pickers_area[area_id] if picker.rent == "short"])
             for area_id in env.area_ids],
            [0, 0, 0],
        )

    def test_fixed_hybrid_requires_at_least_two_decision_days(self):
        from agent.training_utils import max_decisions

        with self.assertRaisesRegex(ValueError, "max_days >= 2"):
            max_decisions("fixed_hybrid", 1)


if __name__ == "__main__":
    unittest.main()
