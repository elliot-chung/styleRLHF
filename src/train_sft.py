"""
Phase 1: Supervised Fine-Tuning.
Train the policy to predict the masked item given an outfit with one slot masked.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import (
    OUTPUT_DIR,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LR,
    NUM_SLOTS,
    NUM_CHUNKS,
    SLOTS,
)
from .data import Catalog, build_synthetic_outfit_dataset, create_sft_examples
from .models import PolicyNetwork


class SFTDataset(Dataset):
    def __init__(self, examples, catalog: Catalog):
        self.examples = examples
        self.catalog = catalog

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_vec, slot_index, target_item_id = self.examples[i]
        valid_ids = self.catalog.item_ids_for_slot(SLOTS[slot_index])
        return {
            "outfit": torch.tensor(input_vec, dtype=torch.long),
            "slot_index": slot_index,
            "target": target_item_id,
            "valid_ids": valid_ids,
        }


def collate_sft(batch, vocab_size, device):
    outfits = torch.stack([b["outfit"] for b in batch])
    slot_indices = torch.tensor([b["slot_index"] for b in batch], dtype=torch.long, device=device)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long, device=device)
    valid_ids_per_slot = [b["valid_ids"] for b in batch]
    mask = torch.zeros(len(batch), vocab_size + 1, dtype=torch.bool, device=device)
    for b in range(len(batch)):
        for iid in valid_ids_per_slot[b]:
            if iid <= vocab_size:
                mask[b, iid] = True
    return {
        "outfit": outfits.to(device),
        "slot_index": slot_indices,
        "target": targets,
        "action_mask": mask,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=None, help="Path to catalog CSV")
    parser.add_argument("--num-outfits", type=int, default=5000, help="Number of synthetic outfits")
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=SFT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=SFT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Path to save SFT model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus_path = Path(args.corpus) if args.corpus else None
    catalog = Catalog()
    if corpus_path and corpus_path.exists():
        catalog.load_csv(corpus_path)
    else:
        raise FileNotFoundError(f"Catalog CSV not found at {corpus_path}. Run scripts/prepare_corpus.py to create data.")

    if catalog.num_items == 0:
        raise RuntimeError("Catalog is empty. Run scripts/prepare_corpus.py to create data.")

    rng = random.Random(args.seed)
    full_outfits = build_synthetic_outfit_dataset(catalog, args.num_outfits, rng, style_coherent=True)
    chunk_size = int(args.num_outfits // NUM_CHUNKS)
    examples = []
    ind = 0
    for num_mask in range(1, NUM_SLOTS):
        num_chunks = NUM_SLOTS - num_mask
        offset = num_chunks * chunk_size
        chunk = full_outfits[ind:ind+offset]
        examples.extend(create_sft_examples(chunk, catalog, rng, num_mask=num_mask))
        ind += offset
    vocab_size = catalog.num_items 

    dataset = SFTDataset(examples, catalog)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, vocab_size, device),
    )

    model = PolicyNetwork(vocab_size=vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            logits = model(
                batch["outfit"],
                batch["slot_index"],
                action_mask=batch["action_mask"],
            )
            loss = criterion(logits, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

    save_path = Path(args.save) if args.save else OUTPUT_DIR / "sft_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
    }, save_path)
    print(f"Saved SFT model to {save_path}")


if __name__ == "__main__":
    main()
