"""
Phase 3: PPO (simplified) to maximize reward while staying close to SFT policy.
We use a single-step MDP: state = incomplete outfit, action = one item for first empty slot.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .config import (
    OUTPUT_DIR,
    CORPUS_CSV,
    PPO_BATCH_SIZE,
    PPO_EPOCHS,
    PPO_KL_COEF,
    PPO_LR,
    PPO_STEPS,
    NUM_SLOTS,
    SLOTS,
    EMPTY_TOKEN_ID,
)
from .data import Catalog, get_random_incomplete_outfit
from .environment import OutfitEnvironment
from .models import PolicyNetwork, RewardModel


def ppo_step(
    policy: PolicyNetwork,
    ref_policy: PolicyNetwork,
    reward_model: RewardModel,
    batch_outfits: list,
    catalog: Catalog,
    env: OutfitEnvironment,
    device: torch.device,
    kl_coef: float,
    opt: torch.optim.Optimizer,
) -> float:
    """One PPO step: get actions, rewards, KL, and update policy."""
    batch_size = len(batch_outfits)
    outfit_t = torch.tensor(batch_outfits, dtype=torch.long, device=device)
    slot_indices = []
    valid_ids_per_slot = []
    for outfit in batch_outfits:
        si = env.get_first_empty_slot(outfit)
        slot_indices.append(si if si is not None else 0)
        valid_ids_per_slot.append(catalog.item_ids_for_slot(SLOTS[slot_indices[-1]]) if slot_indices[-1] is not None else [])
    slot_t = torch.tensor(slot_indices, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, policy.vocab_size + 1, dtype=torch.bool, device=device)
    for b in range(batch_size):
        for iid in valid_ids_per_slot[b]:
            if iid <= policy.vocab_size:
                mask[b, iid] = True

    # Sample action from current policy
    with torch.no_grad():
        action, old_log_prob = policy.sample(outfit_t, slot_t, mask, temperature=1.0)
    # Completed outfits (state after action)
    completed = []
    for b in range(batch_size):
        out = list(batch_outfits[b])
        idx = slot_indices[b]
        if idx is not None:
            out[idx] = action[b].item()
        completed.append(out)

    comp_t = torch.tensor(completed, dtype=torch.long, device=device)
    with torch.no_grad():
        reward = reward_model(comp_t)
        ref_log_prob = ref_policy.get_log_probs(outfit_t, slot_t, action, mask)

    log_prob = policy.get_log_probs(outfit_t, slot_t, action, mask)
    ratio = torch.exp(log_prob - old_log_prob.detach())
    kl = (old_log_prob - ref_log_prob).mean()
    surr = (ratio * reward).mean()
    loss = -surr + kl_coef * kl
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    opt.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", type=str, default=None)
    parser.add_argument("--reward-model", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--steps", type=int, default=PPO_STEPS)
    parser.add_argument("--epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PPO_LR)
    parser.add_argument("--kl-coef", type=float, default=PPO_KL_COEF)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    catalog = Catalog()
    corpus_path = Path(args.corpus) if args.corpus else CORPUS_CSV
    if corpus_path.exists():
        catalog.load_csv(corpus_path)
    else:
        raise FileNotFoundError(f"Catalog CSV not found at {args.corpus}. Run scripts/prepare_corpus.py to create data.")

    sft_path = Path(args.sft_model) if args.sft_model else OUTPUT_DIR / "sft_model.pt"
    rm_path = Path(args.reward_model) if args.reward_model else OUTPUT_DIR / "reward_model.pt"
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT model not found: {sft_path}")
    if not rm_path.exists():
        raise FileNotFoundError(f"Reward model not found: {rm_path}. Run train_reward.py (or collect_preferences first).")

    ckpt = torch.load(sft_path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    policy = PolicyNetwork(vocab_size=vocab_size).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    rm_ckpt = torch.load(rm_path, map_location=device)
    reward_model = RewardModel(vocab_size=rm_ckpt["vocab_size"]).to(device)
    reward_model.load_state_dict(rm_ckpt["model_state_dict"])
    reward_model.eval()

    env = OutfitEnvironment(catalog)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    rng = random.Random(args.seed) 

    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(range(args.steps), desc=f"PPO Epoch {epoch+1}")
        for _ in pbar:
            batch = [get_random_incomplete_outfit(catalog, min_empty=1, max_empty=NUM_SLOTS - 1, rng=rng) for _ in range(args.batch_size)]
            loss = ppo_step(policy, ref_policy, reward_model, batch, catalog, env, device, args.kl_coef, opt)
            total_loss += loss
            n_steps += 1
            pbar.set_postfix(loss=loss)
        print(f"PPO Epoch {epoch+1} avg loss: {total_loss / max(n_steps, 1):.4f}")

    save_path = Path(args.save) if args.save else OUTPUT_DIR / "ppo_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict(), "vocab_size": vocab_size}, save_path)
    print(f"Saved PPO policy to {save_path}")


if __name__ == "__main__":
    main()
