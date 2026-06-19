"""Synthetic outfit generation and SFT example creation.

We have no human-labeled "perfect" outfits, so we build synthetic full outfits
by sampling one item per slot (optionally constrained to share a style tag).
From those we derive masked SFT examples and random incomplete outfits used as
RL/preference-collection states.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from ..config import SLOTS, NUM_SLOTS, EMPTY_TOKEN_ID
from .catalog import Catalog


def _resolve_rng(rng: Optional[random.Random]) -> random.Random:
    return rng if rng is not None else random


def build_synthetic_outfit_dataset(
    catalog: Catalog,
    num_outfits: int,
    rng: Optional[random.Random] = None,
    style_coherent: bool = True,
) -> List[List[int]]:
    """Build ``num_outfits`` full outfit vectors ``[Top, Bottom, Shoes, Accessory]``.

    When ``style_coherent`` is True, a single style tag is chosen per outfit and
    each slot is sampled from items carrying that tag (falling back to any item
    in the slot when none match).
    """
    rng = _resolve_rng(rng)

    # Style tags available per slot, used to pick a coherent tag per outfit.
    style_tags_per_slot = {
        slot: {
            catalog.get(iid).style_tag
            for iid in catalog.item_ids_for_slot(slot)
        }
        for slot in SLOTS
    }
    # Tags shared across every slot give the best chance at a complete outfit.
    common_tags = set.intersection(*style_tags_per_slot.values()) if style_tags_per_slot else set()
    all_tags = sorted(set().union(*style_tags_per_slot.values())) if style_tags_per_slot else []
    candidate_tags = sorted(common_tags) if common_tags else all_tags

    outfits: List[List[int]] = []
    for _ in range(num_outfits):
        style_tag = rng.choice(candidate_tags) if (style_coherent and candidate_tags) else None
        outfit: List[int] = []
        for slot in SLOTS:
            if style_tag is not None:
                ids = catalog.item_ids_for_slot_style(slot, style_tag)
            else:
                ids = catalog.item_ids_for_slot(slot)
            if not ids:
                outfit.append(EMPTY_TOKEN_ID)
            else:
                outfit.append(rng.choice(ids))
        outfits.append(outfit)
    return outfits


def create_sft_examples(
    outfits: Sequence[Sequence[int]],
    catalog: Catalog,
    rng: Optional[random.Random] = None,
    num_mask: int = 1,
) -> List[Tuple[List[int], int, int]]:
    """Create masked SFT examples from full outfits.

    For each outfit, ``num_mask`` slots are masked (set to ``EMPTY_TOKEN_ID``).
    One example is emitted per masked slot as
    ``(input_vector, slot_index, target_item_id)`` where ``input_vector`` has all
    masked slots zeroed and the target is the slot's original item id.
    """
    rng = _resolve_rng(rng)
    num_mask = max(1, min(num_mask, NUM_SLOTS))

    examples: List[Tuple[List[int], int, int]] = []
    for outfit in outfits:
        # Only mask slots that are actually filled.
        fillable = [i for i in range(NUM_SLOTS) if outfit[i] != EMPTY_TOKEN_ID]
        if not fillable:
            continue
        k = min(num_mask, len(fillable))
        masked_slots = rng.sample(fillable, k)
        input_vector = list(outfit)
        for si in masked_slots:
            input_vector[si] = EMPTY_TOKEN_ID
        for si in masked_slots:
            examples.append((list(input_vector), si, outfit[si]))
    return examples


def get_random_incomplete_outfit(
    catalog: Catalog,
    min_empty: int = 1,
    max_empty: int = NUM_SLOTS - 1,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Return a full outfit with a random number of slots emptied.

    Builds one style-coherent full outfit then zeros ``k`` randomly chosen slots,
    where ``k`` is sampled from ``[min_empty, max_empty]``.
    """
    rng = _resolve_rng(rng)
    full = build_synthetic_outfit_dataset(catalog, 1, rng=rng, style_coherent=True)[0]

    lo = max(0, min_empty)
    hi = min(max_empty, NUM_SLOTS)
    hi = max(hi, lo)
    k = rng.randint(lo, hi)

    empty_slots = rng.sample(range(NUM_SLOTS), k) if k > 0 else []
    outfit = list(full)
    for si in empty_slots:
        outfit[si] = EMPTY_TOKEN_ID
    return outfit
