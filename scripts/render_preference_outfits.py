"""
Render outfit images for a single preference pair from a dataset_preferences.json file.
Usage: python scripts/render_preference_outfits.py <path_to_json> <index>
Saves outfit_a and outfit_b images to the outputs/ folder.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CORPUS_CSV, OUTPUT_DIR
from src.data import Catalog
from src.environment import OutfitEnvironment


def main():
    parser = argparse.ArgumentParser(
        description="Render both outfits of a preference pair to images in outputs/."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to dataset_preferences.json (or same format).",
    )
    parser.add_argument(
        "index",
        type=int,
        help="Index of the pair in the 'pairs' array (0-based).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"Path to catalog CSV (default: {CORPUS_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save images (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    json_path = args.json_path
    if not json_path.exists():
        sys.exit(f"JSON file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    
    pairs = data.get("pairs", [])
    if not pairs:
        sys.exit("JSON has no 'pairs' array.")

    if args.index < 0 or args.index >= len(pairs):
        sys.exit(f"Index {args.index} out of range [0, {len(pairs) - 1}].")

    pair = pairs[args.index]
    winner = pair["winner"]
    outfit_a = pair["outfit_a"]
    outfit_b = pair["outfit_b"]

    corpus_path = args.corpus or CORPUS_CSV
    if not corpus_path.exists():
        sys.exit(f"Catalog CSV not found: {corpus_path}. Run scripts/prepare_fpi_corpus.py or similar.")

    catalog = Catalog()
    catalog.load_csv(corpus_path)
    env = OutfitEnvironment(catalog)

    img_a = env.render_outfit(outfit_a)
    img_b = env.render_outfit(outfit_b)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path_a = out_dir / f"outfit_{args.index}_a.png"
    path_b = out_dir / f"outfit_{args.index}_b.png"

    img_a.save(path_a)
    img_b.save(path_b)
    print(f"Saved {path_a}")
    print(f"Saved {path_b}")
    print(f"Winner: {winner}")

if __name__ == "__main__":
    main()
