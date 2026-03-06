"""
Collect preference pairs (outfit A, outfit B, winner) using the SFT model and VLM judge.
Saves dataset for reward model training.
Uses async judge calls with rate limiting via the limits library.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import torch
from limits import parse
from limits.aio import storage as aio_storage
from limits.aio import strategies as aio_strategies
from tqdm import tqdm

from .config import (
    OUTPUT_DIR,
    PREFERENCE_NUM_PAIRS,
    VLM_TEMPERATURE,
    NUM_SLOTS,
    SLOTS,
    EMPTY_TOKEN_ID,
    CORPUS_CSV,
    GEMMA_RPM_LIMIT,
    GEMINI_RPM_LIMIT,
    OPENAI_RPM_LIMIT,
    MAX_CONCURRENT_JUDGES,
)
from .data import Catalog, get_random_incomplete_outfit
from .environment import OutfitEnvironment
from .models import PolicyNetwork
from .vlm_judge import judge_pair


def fill_one_slot(
    model: PolicyNetwork,
    outfit: list,
    slot_index: int,
    catalog: Catalog,
    device: torch.device,
    temperature: float = 1.0,
) -> int:
    """Sample one item for the given slot; return item_id (and optionally updated outfit)."""
    valid_ids = catalog.item_ids_for_slot(SLOTS[slot_index])
    if not valid_ids:
        return EMPTY_TOKEN_ID
    vocab_size = catalog.num_items
    mask = torch.zeros(1, vocab_size + 1, dtype=torch.bool, device=device)
    for iid in valid_ids:
        mask[0, iid] = True
    outfit_t = torch.tensor([outfit], dtype=torch.long, device=device)
    slot_t = torch.tensor([slot_index], dtype=torch.long, device=device)
    with torch.no_grad():
        action, _ = model.sample(outfit_t, slot_t, mask, temperature=temperature)
    return action.item()


def complete_outfit(
    model: PolicyNetwork,
    incomplete: list,
    catalog: Catalog,
    env: OutfitEnvironment,
    device: torch.device,
    temperature: float,
) -> list:
    """Fill empty slots one by one (left to right) until no empty slot."""
    outfit = list(incomplete)
    while True:
        slot_index = env.get_first_empty_slot(outfit)
        if slot_index is None:
            break
        item_id = fill_one_slot(model, outfit, slot_index, catalog, device, temperature)
        outfit[slot_index] = item_id
    return outfit


async def _run_async_judges(
    pending: list,
    backend_name: str,
    rpm_limit: int,
    max_concurrent: int,
) -> list:
    """Run judge calls asynchronously with rate limiting via the limits library."""
    limits_storage = aio_storage.MemoryStorage()
    limiter = aio_strategies.MovingWindowRateLimiter(limits_storage)
    rate_limit = parse(f"{rpm_limit}/minute")
    namespace = "vlm_judge"
    identifier = backend_name
    semaphore = asyncio.Semaphore(max_concurrent)

    async def wait_and_consume_limit() -> bool:
        """Wait until under the rate limit, then consume one slot. Returns True when allowed."""
        while True:
            if await limiter.test(rate_limit, namespace, identifier):
                return await limiter.hit(rate_limit, namespace, identifier)
            await asyncio.sleep(0.25)

    async def do_one_judge(comp_a: list, comp_b: list, img_a, img_b):
        async with semaphore:
            await wait_and_consume_limit()
            try:
                winner = await asyncio.to_thread(
                    judge_pair, img_a, img_b, backend=backend_name
                )
                return {"outfit_a": comp_a, "outfit_b": comp_b, "winner": winner}
            except Exception as e:
                print(f"VLM call failed: {e}, skipping pair")
                return None

    tasks = [
        do_one_judge(comp_a, comp_b, img_a, img_b)
        for (comp_a, comp_b, img_a, img_b) in pending
    ]
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Judging pairs",
        unit="pair",
    ):
        result = await coro
        if result is not None:
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", type=str, default=None, help="Path to SFT model .pt")
    parser.add_argument("--corpus", type=str, default=CORPUS_CSV)
    parser.add_argument("--num-pairs", type=int, default=PREFERENCE_NUM_PAIRS)
    parser.add_argument("--temperature", type=float, default=VLM_TEMPERATURE)
    parser.add_argument("--backend", type=str, default="gemma", choices=["openai", "gemini", "gemma"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


async def _main_async(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    catalog = Catalog()
    if args.corpus and Path(args.corpus).exists():
        catalog.load_csv(Path(args.corpus))
    else:
        raise FileNotFoundError(f"Catalog CSV not found at {args.corpus}. Run scripts/prepare_corpus.py to create data.")
        
    print("Num Items", catalog.num_items)

    sft_path = Path(args.sft_model) if args.sft_model else OUTPUT_DIR / "sft_model.pt"
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT model not found: {sft_path}. Run train_sft.py first.")
    ckpt = torch.load(sft_path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    model = PolicyNetwork(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = OutfitEnvironment(catalog)
    rng = random.Random(args.seed)
    pairs = []

    # RPM limits and max workers per backend (parallel path for all)
    _BACKEND_RPM = {"gemma": GEMMA_RPM_LIMIT, "gemini": GEMINI_RPM_LIMIT, "openai": OPENAI_RPM_LIMIT}
    rpm_limit = _BACKEND_RPM[args.backend] - 5 # 5 requests per minute buffer
    interval_sec = 60.0 / rpm_limit
    print(f"Collecting {args.num_pairs} preference pairs ({args.backend}, rate-limited to {rpm_limit} RPM, {MAX_CONCURRENT_JUDGES} concurrent)")
    print(f"  Min interval between request starts: {interval_sec:.2f}s")

    # Phase 1: generate all (comp_a, comp_b, img_a, img_b) without API calls
    pending = []
    for i in tqdm(range(args.num_pairs), desc="Generating outfit pairs", unit="pair"):
        incomplete = get_random_incomplete_outfit(catalog, min_empty=1, max_empty=NUM_SLOTS - 1, rng=rng)
        comp_a = complete_outfit(model, incomplete, catalog, env, device, args.temperature)
        comp_b = complete_outfit(model, incomplete, catalog, env, device, args.temperature)
        img_a = env.render_outfit(comp_a)
        img_b = env.render_outfit(comp_b)
        pending.append((comp_a, comp_b, img_a, img_b))

    # Phase 2: async rate-limited judge calls (limits library + asyncio)
    pairs = await _run_async_judges(
        pending=pending,
        backend_name=args.backend,
        rpm_limit=rpm_limit,
        max_concurrent=MAX_CONCURRENT_JUDGES,
    )
    print(f"Collected {len(pairs)} preference pairs ({args.backend}, rate-limited)")

    out_path = Path(args.output) if args.output else OUTPUT_DIR / "dataset_preferences.json"
    if out_path.exists(): # Append to existing file
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["vocab_size"] == vocab_size: 
            data["pairs"].extend(pairs)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                return # Append then return so new file is not created
        else: # Create new file if the vocab size is different
            files = [item for item in out_path.parent.iterdir() if item.is_file() 
                     and item.name.startswith("dataset_preferences_") 
                     and item.name.endswith(".json")]
            if files:
                last_file = sorted(files)[-1]
                new_file_num = int(last_file.name.split("_")[-1].split(".")[0]) + 1
                new_file_name = f"dataset_preferences_{new_file_num}.json"
                out_path = out_path.parent / new_file_name
            else:
                out_path = out_path.parent / "dataset_preferences_1.json"

    # Create a new file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs, "vocab_size": vocab_size}, f, indent=2)
    print(f"Saved {len(pairs)} preference pairs to {out_path}")


if __name__ == "__main__":
    main()
