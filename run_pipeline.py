#!/usr/bin/env python
"""
Run the full pipeline: prepare corpus -> SFT -> (optional) collect prefs -> train RM -> PPO -> inference.
Without API keys, skip preference collection and train RM on random preferences for demo.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list, cwd: Path = ROOT) -> bool:
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode == 0


def main():
    # 1. Prepare corpus (sample data if no CSV)
    if not (ROOT / "data" / "corpus" / "catalog.csv").exists():
        run([sys.executable, "scripts/prepare_corpus.py"])
    else:
        print("Corpus already exists, skipping prepare_corpus.")

    # 2. SFT
    if not run([sys.executable, "-m", "src.train_sft", "--num-outfits", "2000", "--epochs", "5"]):
        sys.exit(1)

    # 3. Preference collection (requires OPENAI_API_KEY or GOOGLE_API_KEY)
    # If you have no key, we'll skip and create dummy preferences for RM demo
    prefs = ROOT / "outputs" / "dataset_preferences.json"
    if not prefs.exists():
        if run([sys.executable, "-m", "src.collect_preferences", "--num-pairs", "50"]):
            print("Preference collection done.")
        else:
            print("Preference collection failed (missing API key?).")
            sys.exit(1)

    # 4. Train reward model
    if not run([sys.executable, "-m", "src.train_reward", "--epochs", "3"]):
        sys.exit(1)

    # 5. PPO
    if not run([sys.executable, "-m", "src.train_ppo", "--steps", "200", "--epochs", "2"]):
        sys.exit(1)

    # 6. Inference
    run([sys.executable, "-m", "src.inference", "--model", "outputs/ppo_model.pt", "--save-image", "outputs/outfit.png"])
    print("Pipeline done. Check outputs/outfit.png and outputs/.")


if __name__ == "__main__":
    main()
