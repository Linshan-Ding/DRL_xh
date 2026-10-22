"""Regression tests for the fixes requested in code review.

1. PPO selects its reported result with the same deterministic, frozen tail as the baselines.
2. A robot arrives at the pick point chosen when it departed.
"""
from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent.baselines import common as baseline_common
from agent.ppo import train as ppo_train
from agent.ppo.agent import PPOAgent
from agent.training_utils import init_base_env, is_result_candidate
from environment.class_public import load_config
from environment.class_warehouse import Order
from environment.warehouse_env import WarehouseEnv


class SharedTailRuleTest(unittest.TestCase):
    def test_ppo_and_baselines_use_the_same_tail_rule(self):
        self.assertIs(ppo_train.use_deterministic_action, baseline_common.use_deterministic_action)
        self.assertIs(ppo_train.is_result_candidate, baseline_common.is_result_candidate)

    def test_candidates_are_tail_episodes_unless_tail_is_disabled(self):
        self.assertTrue(is_result_candidate(True, 0.1))
        self.assertFalse(is_result_candidate(False, 0.1))
        self.assertTrue(is_result_candidate(False, 0.0))


class PPODeterministicActionTest(unittest.TestCase):
    def test_deterministic_action_is_the_policy_mean(self):
        parameters = load_config()
        env = init_base_env(parameters)
        policy, value = ppo_train.make_networks(env, parameters)
        agent = PPOAgent(policy, value, parameters=parameters, device="cpu")
        state = env.reset([])

        first = agent.select_action(state, deterministic=True)[0]
        second = agent.select_action(state, deterministic=True)[0]
        matrix, scalar = agent._state_to_tensors(state)
        with torch.no_grad():
            mean, _ = agent.policy_old(matrix, scalar)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, mean.cpu().numpy()[0])


def synthetic_orders(days: int = 2, per_day: int = 25, seed: int = 0) -> list[Order]:
    """Small reproducible order set (the committed pickles hold Windows paths)."""
    rng = random.Random(seed)
    items = list(WarehouseEnv().items.values())
    day = int(load_config()["experiment"]["work_seconds_per_day"])
    arrivals = sorted(rng.uniform(0, days * day - 1) for _ in range(days * per_day))
    orders = []
    for index, arrive_time in enumerate(arrivals):
        chosen = rng.sample(items, rng.randint(2, 4))
        due = arrive_time + rng.choice([1800, 3600, 7200])
        orders.append(Order(f"o{index}", chosen, arrive_time=arrive_time, due_time=due))
    return orders


class PPOTailProtocolTest(unittest.TestCase):
    """Short PPO runs on a two-day synthetic order instance."""

    episodes = 10

    def run_ppo(self, tail_fraction: float, out_dir: str):
        parameters = load_config()
        experiment = parameters["experiment"]
        experiment.update(
            mode="short",
            month=1,
            episodes=self.episodes,
            deterministic_tail_fraction=tail_fraction,
            visdom=False,
            device="cpu",
        )
        parameters["paths"]["ppo_output_dir"] = out_dir
        orders = synthetic_orders()

        update_episodes = []
        original_update = PPOAgent.update
        original_load_orders = ppo_train.load_orders
        current_episode = {"value": 0}
        original_make_episode_env = ppo_train.make_episode_env

        def counting_make_episode_env(base_env, episode_orders):
            current_episode["value"] += 1
            return original_make_episode_env(base_env, episode_orders)

        def recording_update(agent, clear_buffer=True):
            update_episodes.append(current_episode["value"])
            return original_update(agent, clear_buffer=clear_buffer)

        ppo_train.load_orders = lambda *_args, **_kwargs: orders
        ppo_train.make_episode_env = counting_make_episode_env
        PPOAgent.update = recording_update
        try:
            ppo_train.train(parameters)
        finally:
            ppo_train.load_orders = original_load_orders
            ppo_train.make_episode_env = original_make_episode_env
            PPOAgent.update = original_update

        stem = ppo_train.case_stem("short", int(experiment["item_scenario"]), 1, int(experiment["seed"]))
        with (Path(out_dir) / f"{stem}_best_config.csv").open(encoding="utf-8") as f:
            best_episodes = {int(row["best_episode"]) for row in csv.DictReader(f)}
        return update_episodes, best_episodes

    def test_tail_episodes_are_frozen_and_the_only_candidates(self):
        with tempfile.TemporaryDirectory() as out_dir:
            update_episodes, best_episodes = self.run_ppo(0.1, out_dir)
        self.assertEqual(update_episodes, list(range(1, self.episodes)))
        self.assertEqual(best_episodes, {self.episodes})

    def test_disabled_tail_keeps_every_episode_trainable(self):
        with tempfile.TemporaryDirectory() as out_dir:
            update_episodes, best_episodes = self.run_ppo(0.0, out_dir)
        self.assertEqual(update_episodes, list(range(1, self.episodes + 1)))
        self.assertEqual(len(best_episodes), 1)


class RobotDepartureTargetTest(unittest.TestCase):
    def test_robot_arrives_at_the_pick_point_chosen_on_departure(self):
        env = WarehouseEnv()
        env.reset([])
        env.adjust_robots_and_pickers(1, {area_id: 0 for area_id in env.area_ids}, first_step=True)
        robot = env.robots[0]

        # Aisle 1 has the smallest x; aisle 5 is nearest to the depot at (18, 0).
        near_item = next(item for item in env.items.values() if item.pick_point_id == "5-1")
        far_item = next(item for item in env.items.values() if item.pick_point_id == "1-1")
        order = Order("o1", [far_item, near_item], arrive_time=0, due_time=3600)
        env.orders_unassigned.append(order)
        env.orders_uncompleted.append(order)

        env.assign_order_to_robot()  # rule 2 (nearest) picks aisle 5 at departure
        target = robot.target_pick_point
        self.assertEqual(target.point_id, "5-1")

        # A rule that would now choose another point must not redirect the robot on arrival.
        robot.pick_point_selection_rule = 1
        env.handle_events(robot.move_to_pick_point_time)

        self.assertIs(robot.pick_point, target)
        self.assertIn(robot, target.robot_queue)
        self.assertIsNone(robot.target_pick_point)


if __name__ == "__main__":
    unittest.main()
