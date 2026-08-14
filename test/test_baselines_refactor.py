from __future__ import annotations

import inspect
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.baselines import a2c, ddpg, ddqn, dqn, sac, td3
from agent.baselines.common import (
    A2CActor,
    BranchingQNetwork,
    DeterministicActor,
    DiscreteReplayBuffer,
    QNetwork,
    ReplayBuffer,
    RunningRewardNormalizer,
    SquashedGaussianActor,
    ValueNetwork,
    discrete_action_cardinalities,
    discrete_indices_to_env_action,
    hard_update,
    linear_exploration_scale,
    max_discrete_action_cardinalities,
    normalized_to_env_action,
    state_to_arrays,
    use_deterministic_action,
    use_random_action,
)
from agent.baselines.train_batch import VALID_ALGORITHMS, build_training_args, month_seed


class BaselineCommonTest(unittest.TestCase):
    def test_long_action_maps_uniformly_to_absolute_resource_bounds(self):
        action = normalized_to_env_action(
            np.asarray([-1.0, 0.0, 1.0, -0.5]),
            action_scale=5,
            mode="long",
            decision_index=0,
            long_robot_max=15,
            long_picker_max=5,
        )
        np.testing.assert_array_equal(action, [1, 3, 5, 2])

    def test_hybrid_only_uses_absolute_mapping_for_first_decision(self):
        normalized = np.asarray([-1.0, 0.0, 1.0, -0.5])
        first = normalized_to_env_action(
            normalized,
            action_scale=5,
            mode="hybrid",
            decision_index=0,
            long_robot_max=15,
            long_picker_max=5,
        )
        later = normalized_to_env_action(
            normalized,
            action_scale=5,
            mode="hybrid",
            decision_index=1,
            long_robot_max=15,
            long_picker_max=5,
        )
        np.testing.assert_array_equal(first, [1, 3, 5, 2])
        np.testing.assert_array_equal(later, [-5, 0, 5, -2])

    def test_state_normalization_matches_ppo_scales(self):
        shape = (9, 20)
        state = {
            "robot_queue_list": np.full(shape, 180, dtype=np.float32),
            "picker_list": np.ones(shape, dtype=np.float32),
            "unpicked_items_list": np.full(shape, 10, dtype=np.float32),
            "n_robots": 180,
            "n_pickers_area": [60, 60, 60],
        }
        matrix, scalar = state_to_arrays(state)
        np.testing.assert_allclose(matrix, 1.0)
        np.testing.assert_allclose(scalar, 1.0)

    def test_learning_and_deterministic_tail_boundaries(self):
        self.assertTrue(use_random_action(63, 64))
        self.assertFalse(use_random_action(64, 64))
        self.assertFalse(use_deterministic_action(900, 1000, 0.1))
        self.assertTrue(use_deterministic_action(901, 1000, 0.1))

    def test_all_baselines_use_configured_ten_percent_tail(self):
        for module in (a2c, ddpg, td3, sac, dqn, ddqn):
            args = module.build_parser().parse_args([])
            self.assertAlmostEqual(args.deterministic_tail_fraction, 0.1)

    def test_exploration_noise_decays_to_zero_before_tail(self):
        self.assertAlmostEqual(linear_exploration_scale(0, 1000, 0.15, 0.0, 0.8), 0.15)
        self.assertAlmostEqual(linear_exploration_scale(800, 1000, 0.15, 0.0, 0.8), 0.0)

    def test_running_reward_normalizer_centers_observed_rewards(self):
        normalizer = RunningRewardNormalizer()
        for reward in (1.0, 2.0, 3.0):
            normalizer.update(reward)
        normalized = normalizer.normalize(torch.tensor([[1.0], [2.0], [3.0]]))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(normalizer.mean, 2.0)
        self.assertAlmostEqual(normalizer.std, math.sqrt(2.0 / 3.0))

    def test_batch_months_receive_distinct_reproducible_seeds(self):
        self.assertEqual(month_seed(0, 1), 0)
        self.assertEqual(month_seed(0, 12), 11)
        self.assertEqual(month_seed(7, 3), 9)

    def test_batch_episode_override_reaches_algorithm(self):
        args = build_training_args(ddpg, mode="long", month=3, episodes=70)
        self.assertEqual(args.mode, "long")
        self.assertEqual(args.month, 3)
        self.assertEqual(args.seed, 2)
        self.assertEqual(args.episodes, 70)

    def test_batch_launcher_accepts_discrete_algorithms(self):
        self.assertIn("dqn", VALID_ALGORITHMS)
        self.assertIn("ddqn", VALID_ALGORITHMS)
        args = build_training_args(ddqn, mode="hybrid", month=4, episodes=80)
        self.assertEqual(args.mode, "hybrid")
        self.assertEqual(args.month, 4)
        self.assertEqual(args.seed, 3)
        self.assertEqual(args.episodes, 80)

    def test_off_policy_learning_starts_after_one_full_batch(self):
        for module in (ddpg, td3, sac, dqn, ddqn):
            args = module.build_parser().parse_args([])
            self.assertEqual(args.learning_starts, 64)
            self.assertEqual(args.batch_size, 64)

    def test_baselines_do_not_save_checkpoints(self):
        for module in (a2c, ddpg, td3, sac, dqn, ddqn):
            source = inspect.getsource(module)
            self.assertNotIn("save_checkpoint", source)
            self.assertNotIn("model_path", source)

    def test_discrete_actions_cover_long_and_short_resource_ranges(self):
        long_counts = discrete_action_cardinalities("long", 0, 4, 5, 15, 5)
        short_counts = discrete_action_cardinalities("short", 0, 4, 5, 15, 5)
        hybrid_counts = max_discrete_action_cardinalities("hybrid", 4, 5, 15, 5)
        np.testing.assert_array_equal(long_counts, [15, 5, 5, 5])
        np.testing.assert_array_equal(short_counts, [11, 11, 11, 11])
        np.testing.assert_array_equal(hybrid_counts, [15, 11, 11, 11])

        long_action = discrete_indices_to_env_action(
            np.asarray([14, 0, 4, 2]), "long", 0, 5, 15, 5
        )
        short_action = discrete_indices_to_env_action(
            np.asarray([0, 5, 10, 3]), "short", 0, 5, 15, 5
        )
        np.testing.assert_array_equal(long_action, [15, 1, 5, 3])
        np.testing.assert_array_equal(short_action, [-5, 0, 5, -2])

    def test_hybrid_discrete_cardinalities_switch_after_first_decision(self):
        first = discrete_action_cardinalities("hybrid", 0, 4, 5, 15, 5)
        later = discrete_action_cardinalities("hybrid", 1, 4, 5, 15, 5)
        np.testing.assert_array_equal(first, [15, 5, 5, 5])
        np.testing.assert_array_equal(later, [11, 11, 11, 11])

    def test_branch_greedy_action_masks_invalid_outputs(self):
        branches = (
            torch.arange(15, dtype=torch.float32).unsqueeze(0),
            torch.arange(11, dtype=torch.float32).unsqueeze(0),
        )
        cardinalities = torch.tensor([[11, 5]])
        action = dqn.greedy_branch_actions(branches, cardinalities)
        np.testing.assert_array_equal(action.numpy(), [[10, 4]])

    def test_ddqn_selects_with_online_network_while_dqn_uses_target_network(self):
        online_q = (torch.tensor([[0.0, 10.0]]),)
        target_q = (torch.tensor([[5.0, 1.0]]),)
        cardinalities = torch.tensor([[2]])
        dqn_action = dqn.select_target_actions(
            online_q,
            target_q,
            cardinalities,
            double_dqn=False,
        )
        ddqn_action = dqn.select_target_actions(
            online_q,
            target_q,
            cardinalities,
            double_dqn=True,
        )
        np.testing.assert_array_equal(dqn_action.numpy(), [[0]])
        np.testing.assert_array_equal(ddqn_action.numpy(), [[1]])


class A2CLongBatchTest(unittest.TestCase):
    def test_single_transition_advantage_is_not_forced_to_zero(self):
        advantage, _ = a2c.compute_gae(
            rewards=[-100.0],
            dones=[True],
            values=torch.tensor([0.25]),
            next_value=torch.zeros(1),
            gamma=1.0,
            gae_lambda=0.95,
            device=torch.device("cpu"),
        )
        self.assertNotEqual(float(advantage.item()), 0.0)

    def test_long_batch_updates_actor_parameters(self):
        torch.manual_seed(0)
        actor = A2CActor(2, 2, scalar_dim=4, action_dim=4)
        value = ValueNetwork(2, 2, scalar_dim=4)
        optimizer = optim.Adam(list(actor.parameters()) + list(value.parameters()), lr=3e-4)
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        matrix = torch.zeros(1, 3, 2, 2)
        scalar = torch.zeros(1, 4)
        rollout = {"log_probs": [], "entropies": [], "values": [], "rewards": [], "dones": []}
        for index in range(64):
            mean, std = actor(matrix, scalar)
            distribution = torch.distributions.Normal(mean, std)
            raw_action = distribution.sample()
            action = torch.tanh(raw_action)
            rollout["log_probs"].append(
                (
                    distribution.log_prob(raw_action)
                    - torch.log(1 - action.pow(2) + 1e-6)
                ).sum(dim=-1).squeeze(0)
            )
            rollout["entropies"].append(distribution.entropy().sum(dim=-1).squeeze(0))
            rollout["values"].append(value(matrix, scalar).squeeze())
            rollout["rewards"].append(float(-(index + 1)))
            rollout["dones"].append(True)

        args = SimpleNamespace(
            gamma=1.0,
            gae_lambda=0.95,
            value_coef=0.5,
            max_grad_norm=0.5,
        )
        a2c.update_a2c(
            actor,
            value,
            optimizer,
            nn.MSELoss(),
            **rollout,
            next_value=torch.zeros(1),
            entropy_coef=0.01,
            args=args,
            device=torch.device("cpu"),
        )
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
        )


class OffPolicyUpdateTest(unittest.TestCase):
    matrix_shape = (3, 2, 2)
    scalar_dim = 4
    action_dim = 4
    device = torch.device("cpu")

    def make_buffer(self):
        buffer = ReplayBuffer(128, self.matrix_shape, self.scalar_dim, self.action_dim)
        normalizer = RunningRewardNormalizer()
        for index in range(64):
            matrix = np.full(self.matrix_shape, index / 64, dtype=np.float32)
            scalar = np.full(self.scalar_dim, index / 64, dtype=np.float32)
            action = np.linspace(-1, 1, self.action_dim, dtype=np.float32)
            reward = float(-(index + 1))
            normalizer.update(reward)
            buffer.add(matrix, scalar, action, reward, matrix, scalar, True)
        return buffer, normalizer

    @staticmethod
    def parameters_changed(before, module):
        return any(
            not torch.equal(old, new)
            for old, new in zip(before, module.parameters())
        )

    def test_ddpg_update_changes_actor(self):
        torch.manual_seed(0)
        actor = DeterministicActor(2, 2, self.scalar_dim, self.action_dim)
        actor_target = DeterministicActor(2, 2, self.scalar_dim, self.action_dim)
        critic = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        critic_target = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        hard_update(actor, actor_target)
        hard_update(critic, critic_target)
        actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
        buffer, normalizer = self.make_buffer()
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        args = SimpleNamespace(batch_size=64, gamma=1.0, tau=0.005, max_grad_norm=0.5)
        ddpg.update_ddpg(
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
            buffer,
            normalizer,
            args,
            self.device,
        )
        self.assertTrue(self.parameters_changed(before, actor))

    def test_td3_update_changes_actor_on_delayed_step(self):
        torch.manual_seed(0)
        actor = DeterministicActor(2, 2, self.scalar_dim, self.action_dim)
        actor_target = DeterministicActor(2, 2, self.scalar_dim, self.action_dim)
        critics = [QNetwork(2, 2, self.scalar_dim, self.action_dim) for _ in range(4)]
        critic1, critic2, critic1_target, critic2_target = critics
        hard_update(actor, actor_target)
        hard_update(critic1, critic1_target)
        hard_update(critic2, critic2_target)
        optimizers = [
            optim.Adam(actor.parameters(), lr=1e-4),
            optim.Adam(critic1.parameters(), lr=1e-3),
            optim.Adam(critic2.parameters(), lr=1e-3),
        ]
        buffer, normalizer = self.make_buffer()
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        args = SimpleNamespace(
            batch_size=64,
            gamma=1.0,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            max_grad_norm=0.5,
        )
        td3.update_td3(
            actor,
            actor_target,
            critic1,
            critic2,
            critic1_target,
            critic2_target,
            *optimizers,
            buffer,
            normalizer,
            args,
            self.device,
            update_step=2,
        )
        self.assertTrue(self.parameters_changed(before, actor))

    def test_sac_update_changes_actor_and_temperature(self):
        torch.manual_seed(0)
        actor = SquashedGaussianActor(2, 2, self.scalar_dim, self.action_dim)
        critic1 = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        critic2 = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        critic1_target = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        critic2_target = QNetwork(2, 2, self.scalar_dim, self.action_dim)
        hard_update(critic1, critic1_target)
        hard_update(critic2, critic2_target)
        actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
        critic1_optimizer = optim.Adam(critic1.parameters(), lr=3e-4)
        critic2_optimizer = optim.Adam(critic2.parameters(), lr=3e-4)
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)
        buffer, normalizer = self.make_buffer()
        actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
        alpha_before = log_alpha.detach().clone()
        args = SimpleNamespace(batch_size=64, gamma=1.0, tau=0.005, max_grad_norm=0.5)
        sac.update_sac(
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
            -float(self.action_dim),
            buffer,
            normalizer,
            args,
            self.device,
        )
        self.assertTrue(self.parameters_changed(actor_before, actor))
        self.assertFalse(torch.equal(alpha_before, log_alpha.detach()))


class DiscreteOffPolicyUpdateTest(unittest.TestCase):
    def make_buffer(self):
        matrix_shape = (3, 2, 2)
        scalar_dim = 4
        action_dim = 4
        buffer = DiscreteReplayBuffer(128, matrix_shape, scalar_dim, action_dim)
        normalizer = RunningRewardNormalizer()
        cardinalities = np.asarray([5, 3, 3, 3], dtype=np.int64)
        for index in range(64):
            matrix = np.full(matrix_shape, index / 64, dtype=np.float32)
            scalar = np.full(scalar_dim, index / 64, dtype=np.float32)
            action = np.asarray(
                [index % 5, index % 3, (index + 1) % 3, (index + 2) % 3],
                dtype=np.int64,
            )
            reward = float(-(index + 1))
            normalizer.update(reward)
            buffer.add(
                matrix,
                scalar,
                action,
                reward,
                matrix,
                scalar,
                cardinalities,
                index % 7 == 0,
            )
        return buffer, normalizer

    def test_dqn_and_ddqn_updates_change_online_network(self):
        for double_dqn in (False, True):
            with self.subTest(double_dqn=double_dqn):
                torch.manual_seed(0)
                online = BranchingQNetwork(2, 2, 4, [5, 3, 3, 3])
                target = BranchingQNetwork(2, 2, 4, [5, 3, 3, 3])
                hard_update(online, target)
                optimizer = optim.Adam(online.parameters(), lr=3e-4)
                buffer, normalizer = self.make_buffer()
                before = [parameter.detach().clone() for parameter in online.parameters()]
                args = SimpleNamespace(batch_size=64, gamma=1.0, max_grad_norm=0.5)

                loss = dqn.update_dqn(
                    online,
                    target,
                    optimizer,
                    buffer,
                    normalizer,
                    args,
                    torch.device("cpu"),
                    double_dqn=double_dqn,
                )

                self.assertTrue(math.isfinite(loss))
                self.assertTrue(
                    any(
                        not torch.equal(old, new)
                        for old, new in zip(before, online.parameters())
                    )
                )


if __name__ == "__main__":
    unittest.main()
