from __future__ import annotations

import torch
import torch.nn as nn

from agent.training_utils import layer_init


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 21,
        input_width: int = 10,
        scalar_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 4,
        feature_dim: int = 32,
        attention_heads: int = 4,
        initial_log_std: float = 0.5,
        min_log_std: float = -1.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, feature_dim)),
            nn.ReLU(),
        )
        self.fc_backbone = nn.Sequential(
            layer_init(nn.Linear(feature_dim + scalar_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.action_mean = layer_init(nn.Linear(hidden_dim, output_dim), std=0.01)
        initial_log_std = max(min(initial_log_std, max_log_std), min_log_std)
        self.action_log_std = nn.Parameter(torch.ones(output_dim) * initial_log_std)

    def clamp_action_log_std_(self) -> None:
        with torch.no_grad():
            self.action_log_std.clamp_(self.min_log_std, self.max_log_std)

    def forward(self, matrix_inputs: torch.Tensor, scalar_inputs: torch.Tensor):
        cnn_feat = self.cnn(matrix_inputs)
        visual_feat = self.visual_fc(cnn_feat)
        hidden = self.fc_backbone(torch.cat([visual_feat, scalar_inputs], dim=1))
        attn_input = hidden.unsqueeze(1)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        hidden = self.attn_norm(hidden + attn_out.squeeze(1))
        mean = self.action_mean(hidden)
        log_std = self.action_log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.expand_as(mean).exp()
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 21,
        input_width: int = 10,
        scalar_dim: int = 4,
        hidden_dim: int = 128,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, feature_dim)),
            nn.ReLU(),
        )
        self.fc_backbone = nn.Sequential(
            layer_init(nn.Linear(feature_dim + scalar_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, matrix_inputs: torch.Tensor, scalar_inputs: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(matrix_inputs)
        visual_feat = self.visual_fc(cnn_feat)
        hidden = self.fc_backbone(torch.cat([visual_feat, scalar_inputs], dim=1))
        return self.value_head(hidden)
