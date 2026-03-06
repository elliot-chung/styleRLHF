"""
Generate a full outfit from an empty or partially filled vector.
Use the trained policy to fill one slot at a time until complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import NUM_SLOTS, SLOTS, EMPTY_TOKEN_ID, OUTPUT_DIR, CORPUS_CSV
from .data import Catalog
from .environment import OutfitEnvironment
from .models import PolicyNetwork


def generate_outfit(
    model: PolicyNetwork,
    catalog: Catalog,
    env: OutfitEnvironment,
    device: torch.device,
    start_outfit: list | None = None,
    temperature: float = 0.7,
) -> list:
    """
    Start from start_outfit (or all zeros) and fill empty slots left-to-right.
    Returns the completed outfit vector.
    """
    if start_outfit is None:
        outfit = [EMPTY_TOKEN_ID] * NUM_SLOTS
    else:
        outfit = list(start_outfit)
    vocab_size = model.vocab_size
    while True:
        slot_index = env.get_first_empty_slot(outfit)
        if slot_index is None:
            break
        valid_ids = catalog.item_ids_for_slot(SLOTS[slot_index])
        if not valid_ids:
            break
        mask = torch.zeros(1, vocab_size + 1, dtype=torch.bool, device=device)
        for iid in valid_ids:
            mask[0, iid] = True
        outfit_t = torch.tensor([outfit], dtype=torch.long, device=device)
        slot_t = torch.tensor([slot_index], dtype=torch.long, device=device)
        with torch.no_grad():
            action, _ = model.sample(outfit_t, slot_t, mask, temperature=temperature)
        outfit[slot_index] = action.item()
    return outfit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to SFT or PPO model .pt")
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save-image", type=str, default=None, help="Save rendered outfit to this path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = Catalog()
    corpus_path = Path(args.corpus) if args.corpus else CORPUS_CSV
    if corpus_path.exists():
        catalog.load_csv(corpus_path)
    else:
        raise FileNotFoundError(f"Catalog CSV not found at {args.corpus}. Run scripts/prepare_corpus.py to create data.")

    model_path = Path(args.model) if args.model else OUTPUT_DIR / "ppo_model.pt"
    if not model_path.exists():
        model_path = OUTPUT_DIR / "sft_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("No model found. Run train_sft.py or train_ppo.py first.")
    ckpt = torch.load(model_path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    model = PolicyNetwork(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = OutfitEnvironment(catalog)
    outfit = generate_outfit(model, catalog, env, device, start_outfit=None, temperature=args.temperature)
    print("Outfit vector (slot -> item_id):", dict(zip(SLOTS, outfit)))
    for i, slot in enumerate(SLOTS):
        rec = catalog.get(outfit[i])
        if rec:
            print(f"  {slot}: {rec.style_tag} (id={outfit[i]})")
        else:
            print(f"  {slot}: empty")

    if args.save_image:
        img = env.render_outfit(outfit)
        Path(args.save_image).parent.mkdir(parents=True, exist_ok=True)
        img.save(args.save_image)
        print(f"Saved image to {args.save_image}")


if __name__ == "__main__":
    main()
