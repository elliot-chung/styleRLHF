"""Data layer: catalog loading and synthetic outfit generation."""

from .catalog import Catalog, ItemRecord
from .synthetic import (
    build_synthetic_outfit_dataset,
    create_sft_examples,
    get_random_incomplete_outfit,
)

__all__ = [
    "Catalog",
    "ItemRecord",
    "build_synthetic_outfit_dataset",
    "create_sft_examples",
    "get_random_incomplete_outfit",
]
