"""Catalog: load the item corpus CSV and query items by slot/style.

Each valid CSV row becomes one catalog item. Items are assigned sequential
1-indexed token IDs (0 is reserved for the empty slot) which are what the
models embed and what outfit vectors store. The original Fashion Product
Images (FPI) id is retained via ``image_path`` so the environment can load the
correct thumbnail when rendering outfits for the VLM judge.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ItemRecord:
    """A single catalog item."""

    item_id: int           # sequential 1-indexed token id (embedding/action index)
    original_id: str       # original FPI product id (from the CSV item_id column)
    category: str          # outfit slot (e.g. "Top", "Bottom", "Shoes", "Accessory")
    image_path: str        # path relative to DATA_DIR, e.g. "images/12345.jpg"
    style_tag: str         # style label, e.g. "casual_hot", "athletic", "formal"


class Catalog:
    """In-memory catalog of clothing items, queryable by slot and style."""

    def __init__(self) -> None:
        self._by_id: Dict[int, ItemRecord] = {}
        self._by_slot: Dict[str, List[int]] = defaultdict(list)
        self._by_slot_style: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    def load_csv(self, path) -> None:
        """Load items from a catalog CSV with columns:
        ``item_id, category, image_path, style_tag``.

        Rows are read in file order and each is assigned the next sequential
        1-indexed token id.
        """
        path = Path(path)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                original_id = (row.get("item_id") or "").strip()
                category = (row.get("category") or "").strip()
                image_path = (row.get("image_path") or "").strip()
                style_tag = (row.get("style_tag") or "").strip()
                if not category or not image_path:
                    continue
                token_id = len(self._by_id) + 1
                record = ItemRecord(
                    item_id=token_id,
                    original_id=original_id,
                    category=category,
                    image_path=image_path,
                    style_tag=style_tag,
                )
                self._by_id[token_id] = record
                self._by_slot[category].append(token_id)
                self._by_slot_style[(category, style_tag)].append(token_id)

    @property
    def num_items(self) -> int:
        """Total number of valid catalog items (the model vocab size)."""
        return len(self._by_id)

    def item_ids_for_slot(self, category: str) -> List[int]:
        """Token ids of all items belonging to the given slot category."""
        return self._by_slot.get(category, [])

    def item_ids_for_slot_style(self, category: str, style_tag: str) -> List[int]:
        """Token ids for a (slot, style) pair; falls back to the whole slot if empty."""
        ids = self._by_slot_style.get((category, style_tag), [])
        if ids:
            return ids
        return self.item_ids_for_slot(category)

    def get(self, item_id: int) -> Optional[ItemRecord]:
        """Return the record for a token id, or None for empty/unknown ids."""
        return self._by_id.get(item_id)
