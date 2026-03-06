"""
Evaluate reward model agreement with VLM preference labels.

The reward model was trained on preference pairs (outfit_a, outfit_b, winner)
with loss = -log sigmoid(s_winner - s_loser), i.e. it was trained to rank
outfits the same way the VLM judge did when collecting the dataset.

This script:
  1. Loads preference pairs (from dataset_preferences.json or similar).
  2. Scores each outfit with the reward model; RM "predicts" A if score(A) > score(B).
  3. Compares RM predictions to the stored VLM labels (agreement = accuracy).
  4. With --vlm-eval N: generates N brand-new random outfit pairs (SFT model + catalog),
     runs the VLM judge on them, and reports RM vs current VLM agreement on this fresh set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import (
    CORPUS_CSV,
    OUTPUT_DIR,
    NUM_SLOTS,
    GEMMA_RPM_LIMIT,
    GEMINI_RPM_LIMIT,
    OPENAI_RPM_LIMIT,
    MAX_CONCURRENT_JUDGES,
    VLM_TEMPERATURE,
)
from src.data import Catalog, get_random_incomplete_outfit
from src.environment import OutfitEnvironment
from src.inference import generate_outfit
from src.models import RewardModel, PolicyNetwork
from src.vlm_judge import judge_pair


def load_reward_model(path: Path, device: torch.device) -> RewardModel:
    """Load reward model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    model = RewardModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def rm_predict_winner(
    reward_model: RewardModel,
    outfit_a: list[int],
    outfit_b: list[int],
    device: torch.device,
) -> str:
    """
    Return "A" or "B" according to which outfit has higher reward (mirrors training objective).
    """
    a_t = torch.tensor([outfit_a], dtype=torch.long, device=device)
    b_t = torch.tensor([outfit_b], dtype=torch.long, device=device)
    with torch.no_grad():
        s_a = reward_model(a_t).item()
        s_b = reward_model(b_t).item()
    return "A" if s_a > s_b else "B"


def eval_rm_vs_stored_labels(
    pairs: list[dict],
    reward_model: RewardModel,
    device: torch.device,
) -> tuple[float, int, int]:
    """
    Compare RM predictions to stored VLM labels (winner in each pair).
    Returns (accuracy, num_agree, total).
    """
    agree = 0
    for p in pairs:
        pred = rm_predict_winner(
            reward_model,
            p["outfit_a"],
            p["outfit_b"],
            device,
        )
        if pred == p["winner"]:
            agree += 1
    total = len(pairs)
    acc = agree / total if total else 0.0
    return acc, agree, total


def generate_new_outfit_pairs(
    sft_model: PolicyNetwork,
    catalog: Catalog,
    env: OutfitEnvironment,
    device: torch.device,
    num_pairs: int,
    temperature: float,
    rng: random.Random,
) -> list[tuple[list[int], list[int], object, object]]:
    """
    Generate num_pairs new (comp_a, comp_b, img_a, img_b) using SFT model and catalog.
    Same procedure as collect_preferences: random incomplete outfit, complete twice with sampling.
    Returns list of (comp_a, comp_b, img_a, img_b) for VLM judging.
    """
    pending = []
    for _ in range(num_pairs):
        incomplete = get_random_incomplete_outfit(
            catalog, min_empty=1, max_empty=NUM_SLOTS - 1, rng=rng
        )
        comp_a = generate_outfit(
            sft_model, catalog, env, device,
            start_outfit=incomplete, temperature=temperature,
        )
        comp_b = generate_outfit(
            sft_model, catalog, env, device,
            start_outfit=incomplete, temperature=temperature,
        )
        img_a = env.render_outfit(comp_a)
        img_b = env.render_outfit(comp_b)
        pending.append((comp_a, comp_b, img_a, img_b))
    return pending


async def run_vlm_on_pending(
    pending: list[tuple[list[int], list[int], object, object]],
    backend: str,
    rpm_limit: int,
    max_concurrent: int,
) -> list[dict]:
    """
    Run VLM judge on each (comp_a, comp_b, img_a, img_b) in pending.
    Returns list of {"outfit_a", "outfit_b", "vlm_winner"}.
    """
    try:
        from limits import parse
        from limits.aio import storage as aio_storage
        from limits.aio import strategies as aio_strategies
    except ImportError:
        raise ImportError("For --vlm-eval install 'limits': pip install limits")

    limits_storage = aio_storage.MemoryStorage()
    limiter = aio_strategies.MovingWindowRateLimiter(limits_storage)
    rate_limit = parse(f"{rpm_limit}/minute")
    namespace = "vlm_judge"
    identifier = backend
    semaphore = asyncio.Semaphore(max_concurrent)

    async def wait_and_consume():
        while True:
            if await limiter.test(rate_limit, namespace, identifier):
                return await limiter.hit(rate_limit, namespace, identifier)
            await asyncio.sleep(0.25)

    async def do_one(comp_a: list, comp_b: list, img_a, img_b):
        async with semaphore:
            await wait_and_consume()
            try:
                vlm_winner = await asyncio.to_thread(
                    judge_pair, img_a, img_b, backend=backend
                )
                return {"outfit_a": comp_a, "outfit_b": comp_b, "vlm_winner": vlm_winner}
            except Exception as e:
                print(f"VLM call failed: {e}")
                return None

    tasks = [do_one(ca, cb, ia, ib) for (ca, cb, ia, ib) in pending]
    results = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r is not None:
            results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reward model agreement with VLM preference labels."
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        default=OUTPUT_DIR / "dataset_preferences.json",
        help="Path to preference dataset JSON (outfit_a, outfit_b, winner).",
    )
    parser.add_argument(
        "--reward-model",
        type=Path,
        default=OUTPUT_DIR / "reward_model.pt",
        help="Path to reward model checkpoint.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_CSV,
        help="Path to catalog CSV (required for --vlm-eval).",
    )
    parser.add_argument(
        "--sft-model",
        type=Path,
        default=OUTPUT_DIR / "sft_model.pt",
        help="Path to SFT policy for generating outfits (required for --vlm-eval).",
    )
    parser.add_argument(
        "--vlm-eval",
        type=int,
        default=0,
        metavar="N",
        help="Generate N new random outfit pairs, run VLM judge, report RM vs VLM agreement (requires API).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=VLM_TEMPERATURE,
        help="Sampling temperature for outfit generation in --vlm-eval (default from config).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemma",
        choices=["openai", "gemini", "gemma"],
        help="VLM backend for --vlm-eval (default: gemma).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --vlm-eval outfit generation.",
    )
    args = parser.parse_args()

    if not args.reward_model.exists():
        sys.exit(f"Reward model not found: {args.reward_model}. Run train_reward first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = load_reward_model(args.reward_model, device)

    # 1) RM vs stored VLM labels (same task the RM was trained for) — only if preferences file exists
    if args.preferences.exists():
        with open(args.preferences, encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", [])
        if pairs:
            acc, agree, total = eval_rm_vs_stored_labels(pairs, reward_model, device)
            print("Reward model vs stored VLM labels (training-time task)")
            print(f"  Pairs: {total}")
            print(f"  RM agrees with stored winner: {agree}/{total} = {acc:.2%}")
        else:
            print("Preferences file has no pairs; skipping stored-labels eval.")
    else:
        if args.vlm_eval <= 0:
            sys.exit(f"Preferences file not found: {args.preferences}. Run collect_preferences or use --vlm-eval N.")
        print("No preferences file; skipping stored-labels eval.")

    # 2) --vlm-eval N: generate N new random outfit pairs, run VLM, report RM vs VLM
    if args.vlm_eval > 0:
        if not args.corpus.exists():
            sys.exit(f"Corpus not found for --vlm-eval: {args.corpus}")
        if not args.sft_model.exists():
            sys.exit(f"SFT model not found for --vlm-eval: {args.sft_model}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)

        catalog = Catalog()
        catalog.load_csv(args.corpus)
        env = OutfitEnvironment(catalog)
        ckpt = torch.load(args.sft_model, map_location=device)
        vocab_size = ckpt["vocab_size"]
        sft_model = PolicyNetwork(vocab_size=vocab_size).to(device)
        sft_model.load_state_dict(ckpt["model_state_dict"])
        sft_model.eval()

        n = args.vlm_eval
        print(f"\nGenerating {n} new random outfit pairs (SFT + catalog, temp={args.temperature})...")
        pending = generate_new_outfit_pairs(
            sft_model, catalog, env, device, n, args.temperature, rng
        )
        rpm = {"gemma": GEMMA_RPM_LIMIT - 5, "gemini": GEMINI_RPM_LIMIT - 5, "openai": OPENAI_RPM_LIMIT - 5}[
            args.backend
        ]
        print(f"Running VLM judge ({args.backend}) on {n} pairs...")
        results = asyncio.run(
            run_vlm_on_pending(pending, args.backend, rpm, MAX_CONCURRENT_JUDGES)
        )
        if not results:
            print("  No successful VLM responses.")
        else:
            rm_agree = sum(
                1 for r in results
                if rm_predict_winner(reward_model, r["outfit_a"], r["outfit_b"], device) == r["vlm_winner"]
            )
            n_ok = len(results)
            print(f"  RM agrees with VLM (fresh validation set): {rm_agree}/{n_ok} = {rm_agree/n_ok:.2%}")


if __name__ == "__main__":
    main()
