"""Outfit environment: valid actions per slot and rendering outfit images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np

from .config import SLOTS, NUM_SLOTS, EMPTY_TOKEN_ID, DATA_DIR
from .data import Catalog


class OutfitEnvironment:
    """Handles valid actions for a slot and rendering an outfit as a single image."""

    def __init__(self, catalog: Catalog, image_size: Tuple[int, int] = (60, 80)):
        self.catalog = catalog
        self.image_size = image_size

    def get_valid_actions(self, current_outfit: List[int], target_slot_index: int) -> List[int]:
        """Return list of item IDs that belong to the target slot category."""
        category = SLOTS[target_slot_index]
        return self.catalog.item_ids_for_slot(category)

    def get_first_empty_slot(self, outfit: List[int]) -> Optional[int]:
        """Index of first empty slot, or None if full."""
        for i in range(NUM_SLOTS):
            if outfit[i] == EMPTY_TOKEN_ID:
                return i
        return None

    def render_outfit(
        self,
        outfit: List[int],
        background: str = "white",
    ) -> Image.Image:
        """
        Stitch item images into a single grid (next largest square of len(outfit))
        """
        n = len(outfit)
        size = int(np.ceil(np.sqrt(n)))
        out = Image.new("RGB", (size * self.image_size[0], size * self.image_size[1]), background)
        for i in range(n):
            item_id = outfit[i]
            if item_id == EMPTY_TOKEN_ID:
                tile = Image.new("RGB", (self.image_size[0], self.image_size[1]), (220, 220, 220))
            else:
                rec = self.catalog.get(item_id)
                if rec:
                    path = DATA_DIR / rec.image_path
                    if path.exists():
                        tile = Image.open(path).convert("RGB").resize((self.image_size[0], self.image_size[1]), Image.Resampling.LANCZOS)
                    else:
                        raise FileNotFoundError(f"Image not found: {path}")
                else:
                    raise ValueError(f"Item record not found: {item_id}")
            out.paste(tile, (i % size * self.image_size[0], i // size * self.image_size[1]))
        return out

    def render_outfit_pair_side_by_side(
        self,
        outfit_a: List[int],
        outfit_b: List[int],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Image.Image:
        """Render two outfits side by side for VLM comparison (A | B)."""
        img_a = self.render_outfit(outfit_a)
        img_b = self.render_outfit(outfit_b)
        w1, h1 = img_a.size
        w2, h2 = img_b.size
        gap = 20
        total_w = w1 + gap + w2
        total_h = max(h1, h2) + 30
        out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        out.paste(img_a, (0, 20))
        out.paste(img_b, (w1 + gap, 20))
        # Optional: draw labels with PIL ImageDraw if needed
        return out
