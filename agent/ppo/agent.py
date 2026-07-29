from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agent.ppo.buffer import RolloutBuffer
from environment.class_public import Config


class PPOAgent(Config):
    def __init__(self, policy_network, value_network, parameters=None, device: str | None = None):
        super().__init__()
        if parameters is not None:
            self.parameters = parameters

        configured_device = device or self.parameters["experiment"]["device"]
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(configured_device)

        self.policy = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)
        self.policy_old = copy.deepcopy(policy_network).to(self.device)
        self.policy_old.eval()

        ppo = self.parameters["ppo"]
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": ppo["learning_rate"]},
                {"params": self.value_network.parameters(), "lr": ppo["learning_rate"]},
            ]
        )
        self.gamma = ppo["gamma"]
        self.eps_clip = ppo["clip_range"]
        self.k_epochs = ppo["n_epochs"]
        self.gae_lambda = ppo["gae_lambda"]
        self.batch_size = ppo["batch_size"]
        self.ent_coef = ppo["initial_entropy_coeff"]
        self.ent_coef_decay = ppo["entropy_coeff_decay"]
        self.min_ent_coef = ppo["min_entropy_coeff"]
        self.max_grad_norm = ppo["max_grad_norm"]
        self.value_loss_coeff = ppo["value_loss_coeff"]
        self.normalize_rewards = ppo["normalize_rewards"]
        self.standardize_rewards = ppo["standardize_rewards"]
        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.last_update_stats = {}

        warehouse = self.parameters["warehouse"]
        order = self.parameters["order"]
        pick_points_total = max(
            1,
            int(warehouse["area_num"])
            * int(warehouse["aisle_num"])
            * int(warehouse["shelf_capacity"]),
        )
        pick_points_per_area = max(1, int(warehouse["aisle_num"]) * int(warehouse["shelf_capacity"]))
        max_order_items = max(1, int(max(order["order_n_items"])))
        self.matrix_input_scale = torch.as_tensor(
            [pick_points_total, 1.0, max_order_items],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.scalar_input_scale = torch.as_tensor(
            [pick_points_total] + [pick_points_per_area] * int(warehouse["area_num"]),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self._clamp_policy_log_std()

    def _clamp_policy_log_std(self) -> None:
        for policy in (self.policy, self.policy_old):
            clamp = getattr(policy, "clamp_action_log_std_", None)
            if clamp is not None:
                clamp()

    def _normalize_matrix_inputs(self, matrix_inputs: torch.Tensor) -> torch.Tensor:
        return matrix_inputs / self.matrix_input_scale

    def _normalize_scalar_inputs(self, scalar_inputs: torch.Tensor) -> torch.Tensor:
        return scalar_inputs / self.scalar_input_scale

    def _state_to_tensors(self, state):
        matrix_inputs = torch.as_tensor(
            np.array(
                [
                    state["robot_queue_list"],
                    state["picker_list"],
                    state["unpicked_items_list"],
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        scalar_inputs = torch.as_tensor(
            np.array([state["n_robots"]] + state["n_pickers_area"]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return (
            self._normalize_matrix_inputs(matrix_inputs),
            self._normalize_scalar_inputs(scalar_inputs),
        )

    def _preprocess_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.standardize_rewards:
            centered = rewards - rewards.mean()
            std = rewards.std(unbiased=False)
            if std > 1e-8:
                return centered / (std + 1e-8)
            return centered
        if self.normalize_rewards:
            scale = rewards.abs().max()
            if scale > 1e-8:
                return rewards / scale
        return rewards

    def select_action(self, state):
        self.policy_old.eval()
        with torch.no_grad():
            self._clamp_policy_log_std()
            matrix_inputs, scalar_inputs = self._state_to_tensors(state)
            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            value = self.value_network(matrix_inputs, scalar_inputs)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0],
            matrix_inputs,
            scalar_inputs,
        )

    def update(self, clear_buffer: bool = True) -> bool:
        if len(self.buffer) <= 1:
            if clear_buffer:
                self.buffer.clear()
            return False

        matrix_states = torch.cat(self.buffer.matrix_states).to(self.device)
        scalar_states = torch.cat(self.buffer.scalar_states).to(self.device)
        actions = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_logprobs = torch.as_tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=self.device)
        raw_rewards = torch.as_tensor(np.array(self.buffer.rewards), dtype=torch.float32, device=self.device)
        rewards = self._preprocess_rewards(raw_rewards)
        dones = torch.as_tensor(np.array(self.buffer.dones), dtype=torch.float32, device=self.device)
        values = torch.as_tensor(np.array(self.buffer.values), dtype=torch.float32, device=self.device).squeeze()
        if values.dim() == 0:
            values = values.unsqueeze(0)

        returns = []
        advantages = []
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = torch.zeros((), dtype=torch.float32, device=self.device)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]

        returns = torch.stack(returns).to(self.device)
        advantages = torch.stack(advantages).to(self.device)
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        dataset_size = len(rewards)
        batch_size = min(self.batch_size, dataset_size)
        policy_losses = []
        value_losses = []
        entropies = []
        action_stds = []
        for _ in range(self.k_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)
            for indices in sampler:
                indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                b_matrix = matrix_states[indices]
                b_scalar = scalar_states[indices]
                b_actions = actions[indices]
                b_old_logprobs = old_logprobs[indices]
                b_returns = returns[indices].view(-1)
                b_advantages = advantages[indices]

                mean, std = self.policy(b_matrix, b_scalar)
                dist = Normal(mean, std)
                logprobs = dist.log_prob(b_actions).sum(dim=1)
                dist_entropy = dist.entropy().sum(dim=1)
                state_values = self.value_network(b_matrix, b_scalar).squeeze()
                if state_values.dim() == 0:
                    state_values = state_values.unsqueeze(0)

                ratios = torch.exp(logprobs - b_old_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(state_values, b_returns)
                entropy = dist_entropy.mean()
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._clamp_policy_log_std()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                action_stds.append(float(std.mean().detach().cpu()))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self._clamp_policy_log_std()
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)
        reward_std = raw_rewards.std(unbiased=False) if raw_rewards.numel() > 1 else torch.zeros_like(raw_rewards[0])
        self.last_update_stats = {
            "mean_reward": float(raw_rewards.mean().detach().cpu()),
            "std_reward": float(reward_std.detach().cpu()),
            "mean_return": float(returns.mean().detach().cpu()),
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_action_std": float(np.mean(action_stds)) if action_stds else 0.0,
        }
        if clear_buffer:
            self.buffer.clear()
        return True
