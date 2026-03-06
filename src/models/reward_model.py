"""Reward model: outfit vector -> scalar style score (trained on VLM preferences)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NUM_SLOTS, RM_EMBED_DIM, RM_HIDDEN, SFT_DROPOUT


class RewardModel(nn.Module):
    """Maps full outfit vector to a scalar reward (0-1 style quality)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = RM_EMBED_DIM,
        hidden: int = RM_HIDDEN,
        dropout: float = SFT_DROPOUT,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(NUM_SLOTS * embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, outfit: torch.Tensor) -> torch.Tensor:
        """
        outfit: (B, NUM_SLOTS) long
        Returns: (B,) scalar in [0, 1]
        """
        B = outfit.size(0)
        x = self.embed(outfit)  # (B, NUM_SLOTS, EMBED)
        x = x.view(B, -1)  # (B, NUM_SLOTS*EMBED)
        return self.mlp(x).squeeze(-1)
