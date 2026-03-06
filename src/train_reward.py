"""
Train the reward model on preference data (outfit_a, outfit_b, winner).
RM learns to output a higher score for the preferred outfit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import OUTPUT_DIR, RM_BATCH_SIZE, RM_EPOCHS, RM_LR
from .models import RewardModel


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list, vocab_size: int):
        self.pairs = pairs
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        a = torch.tensor(p["outfit_a"], dtype=torch.long)
        b = torch.tensor(p["outfit_b"], dtype=torch.long)
        winner = p["winner"]
        return a, b, winner


def collate_pref(batch):
    a = torch.stack([b[0] for b in batch])
    b = torch.stack([b[1] for b in batch])
    winner = [b[2] for b in batch]
    return a, b, winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferences", type=str, default=None, help="Path to dataset_preferences.json")
    parser.add_argument("--epochs", type=int, default=RM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=RM_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=RM_LR)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    prefs_path = Path(args.preferences) if args.preferences else OUTPUT_DIR / "dataset_preferences.json"
    if not prefs_path.exists():
        raise FileNotFoundError(f"Preferences not found: {prefs_path}. Run collect_preferences.py first.")
    with open(prefs_path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["pairs"]
    vocab_size = data.get("vocab_size", 100)

    dataset = PreferenceDataset(pairs, vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pref)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel(vocab_size=vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for a, b, winners in tqdm(loader, desc=f"Epoch {epoch+1}"):
            a, b = a.to(device), b.to(device)
            sa = model(a)
            sb = model(b)
            # Preference loss: prefer winner; use log sigmoid(s_winner - s_loser)
            loss = 0.0
            for i in range(a.size(0)):
                if winners[i] == "A":
                    loss = loss + nn.functional.logsigmoid(sa[i] - sb[i])
                else:
                    loss = loss + nn.functional.logsigmoid(sb[i] - sa[i])
            loss = -loss / a.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        print(f"Epoch {epoch+1} avg loss: {total_loss / max(n, 1):.4f}")

    save_path = Path(args.save) if args.save else OUTPUT_DIR / "reward_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size}, save_path)
    print(f"Saved reward model to {save_path}")


if __name__ == "__main__":
    main()
