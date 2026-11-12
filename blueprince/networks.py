"""Neural network parts for PPO (needs torch and stable-baselines3).

BoardCNN reads the same flat observation as the MLP policy, so models are
comparable: the 45 x 17 board block is reshaped into a 17-channel 9 x 5 image
(plus two coordinate channels) and passed through 3x3 convolutions; the global
and blueprint features go through a small dense layer. The two are concatenated
and fed to the usual policy and value heads.
"""
from __future__ import annotations

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .env import BOARD_SIZE, CELL_FEATURES, OBS_SIZE
from .game import COLS, ROWS


class BoardCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, channels: int = 64, layers: int = 3,
                 board_dim: int = 256, extra_dim: int = 64):
        super().__init__(observation_space, features_dim=board_dim + extra_dim)
        convs, in_ch = [], CELL_FEATURES + 2
        for _ in range(layers):
            convs += [nn.Conv2d(in_ch, channels, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = channels
        self.conv = nn.Sequential(*convs)
        self.board_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * ROWS * COLS, board_dim), nn.ReLU())
        self.extra_fc = nn.Sequential(nn.Linear(OBS_SIZE - BOARD_SIZE, extra_dim), nn.ReLU())
        # coordinate channels: zero padding alone does not tell a cell where it is
        rank = th.linspace(0, 1, ROWS).view(ROWS, 1).expand(ROWS, COLS)
        col = th.linspace(0, 1, COLS).view(1, COLS).expand(ROWS, COLS)
        self.register_buffer("coords", th.stack([rank, col]).unsqueeze(0))

    @staticmethod
    def board_image(obs: th.Tensor) -> th.Tensor:
        """(batch, OBS_SIZE) -> (batch, CELL_FEATURES, ROWS, COLS); row 0 is rank 1."""
        b = obs.shape[0]
        return obs[:, :BOARD_SIZE].reshape(b, ROWS, COLS, CELL_FEATURES).permute(0, 3, 1, 2)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        b = obs.shape[0]
        x = th.cat([self.board_image(obs), self.coords.expand(b, -1, -1, -1)], dim=1)
        return th.cat([self.board_fc(self.conv(x)), self.extra_fc(obs[:, BOARD_SIZE:])], dim=1)
