"""Policy network: outfit vector + slot -> distribution over items (with action masking)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import (
    NUM_SLOTS,
    SFT_EMBED_DIM,
    SFT_NHEADS,
    SFT_NLAYERS,
    SFT_DIM_FEEDFORWARD,
    SFT_DROPOUT,
)


class PolicyNetwork(nn.Module):
    """
    Input: (batch, 5) outfit vector (item IDs).
    Output: logits over vocab for a given target slot (action masking applied externally).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = SFT_EMBED_DIM,
        nhead: int = SFT_NHEADS,
        num_layers: int = SFT_NLAYERS,
        dim_feedforward: int = SFT_DIM_FEEDFORWARD,
        dropout: float = SFT_DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = NUM_SLOTS

        self.item_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  # 0 = empty
        self.slot_embed = nn.Embedding(NUM_SLOTS, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size + 1)

    def forward(
        self,
        outfit: torch.Tensor,
        slot_index: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        outfit: (B, 4) long
        slot_index: (B,) long, each in [0, NUM_SLOTS-1]
        action_mask: (B, vocab_size+1) bool, True = allowed. If None, no masking.
        Returns: (B, vocab_size+1) logits
        """
        B = outfit.size(0)
        # (BATCH, NUM_SLOTS, EMBED_DIM)
        x = self.item_embed(outfit)
        # (BATCH, 1, EMBED_DIM) slot embedding and add to the slot position
        slot_emb = self.slot_embed(slot_index)  # (BATCH, EMBED_DIM)
        # Add slot embedding to all positions (or we could add only to the target slot position)
        x = x + slot_emb.unsqueeze(1)
        # (BATCH, NUM_SLOTS, EMBED_DIM) -> transformer
        x = self.transformer(x)  # (BATCH, NUM_SLOTS, EMBED_DIM)
        # Pool: use mean over slots, or use slot position; we use mean for simplicity
        x = x.mean(dim=1)  # (BATCH, EMBED_DIM)
        logits = self.head(x)  # (BATCH, vocab_size+1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits

    def get_log_probs(
        self,
        outfit: torch.Tensor,
        slot_index: torch.Tensor,
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Log probability of the chosen action (B,)"""
        logits = self.forward(outfit, slot_index, action_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def sample(
        self,
        outfit: torch.Tensor,
        slot_index: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        logits = self.forward(outfit, slot_index, action_mask)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


def build_action_mask(
    valid_ids_per_slot: List[List[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    valid_ids_per_slot: list of length B, each element is list of allowed item IDs for that example's slot.
    Returns (B, vocab_size+1) bool, True where action is allowed.
    """
    B = len(valid_ids_per_slot)
    mask = torch.zeros(B, vocab_size + 1, dtype=torch.bool, device=device)
    for b in range(B):
        for iid in valid_ids_per_slot[b]:
            if iid <= vocab_size:
                mask[b, iid] = True
    return mask
