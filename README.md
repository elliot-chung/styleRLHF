# Style RLHF: Outfit Synthesis with Reinforcement Learning from Human Feedback

This project demonstrates **Reinforcement Learning from Human Feedback (RLHF)** in a compact, interpretable setting: training a **Style Agent** that learns to complete outfits by choosing clothing items (Top, Bottom, Shoes, Accessory) from a fixed corpus. The goal is to illustrate the main ideas of RLHF—supervised fine-tuning, reward modeling, and policy optimization—without the scale of full-scale language model alignment.

This README was written with the assistance of LLMs. Any errors are my own. 

---

## Table of Contents

- [Conceptual Overview](#conceptual-overview)
- [ML Concepts Walkthrough](#ml-concepts-walkthrough)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Conceptual Overview

The pipeline has three phases:

1. **SFT (Supervised Fine-Tuning)** — The model learns the *mechanics* of outfit completion: given an outfit with some empty slots, it predicts a valid item for a target slot (e.g. “if I have pants, pick a shirt”). Correctness can be checked programmatically.

2. **Reward modeling** — A separate model (and/or a Vision–Language Model as a proxy for human judgment) learns to score *how good* an outfit looks. We collect preference data (outfit A vs outfit B, which is better?) and train a small **reward model** to predict that preference.

3. **RL (PPO)** — The SFT model is optimized with **Proximal Policy Optimization** to maximize the reward model’s score, while a **KL penalty** keeps the policy close to the SFT policy. This prevents “reward hacking” (e.g. always picking the same item the reward model likes).

By the end, the agent can be given an empty or partial outfit vector and fill slots one-by-one to produce a full, style-coherent outfit.

---

## ML Concepts Walkthrough

### Step 1: State and Action Representation

- **Outfit as a fixed-length vector**  
  An outfit is represented as a vector of **slot indices**: `[Top, Bottom, Shoes, Accessory]`. Each entry is an **item ID** from the corpus (or `0` for “empty”).

- **Example:** `[0, 142, 0, 89]` means “no top, item 142 as bottom, no shoes, item 89 as accessory.”

- **Tokenization:** Each clothing item in the corpus has a unique integer ID (1-indexed). ID `0` is reserved for “empty slot.” The model’s vocabulary size is the number of items plus one.

This gives a discrete, finite state and action space, so we can train with standard supervised and reinforcement learning without dealing with raw pixels in the policy.

### Step 2: Data and Corpus

- **Corpus:** A catalog of items with `item_id`, `category` (slot), `image_path`, and optional `style_tag` (e.g. casual, formal, athletic).

- **Synthetic ground truth:** We don’t have human-labeled “perfect” outfits. So we build **synthetic outfits** by sampling one item per slot, optionally constrained by `style_tag` so that e.g. “athletic” tops are paired with “athletic” bottoms. That yields a set of full outfit vectors we treat as valid.

- **SFT examples:** From each full outfit we **mask** one or more slots (set to 0). Each training example is `(input_vector, target_slot_index, target_item_id)`. The model learns to predict the masked item given the rest of the outfit and the slot to fill.

This mirrors “masked token” pretraining (e.g. BERT), but over outfit slots and with a strict **category constraint**: the model must only predict item IDs that belong to the target slot.

### Step 3: Action Masking (Category Constraint)

- In language models, the next token can be any token. Here, **only items from the target category** are valid (e.g. if the empty slot is “Shoes,” only shoe IDs are allowed).

- **Action masking:** Before applying softmax to the model’s logits, we set logits for all *invalid* item IDs to a large negative value (e.g. `-1e9`). The resulting distribution has mass only on valid items, so sampling and training never choose invalid actions.

The code does this in the policy’s `forward` (and in data loading) by building a boolean `action_mask` from the catalog and applying `masked_fill`.

### Step 4: Phase 1 — Supervised Fine-Tuning (SFT)

- **Goal:** Learn a **policy** that, given an outfit vector and a target slot index, outputs a distribution over item IDs for that slot (with action masking).

- **Model:** A small **Transformer encoder**: outfit vector → item embeddings + slot embedding → transformer layers → mean pooling → linear head → logits over `vocab_size + 1`. Cross-entropy loss is used against the ground-truth item ID.

- **Training:** Batches of `(outfit, slot_index, target_item_id, action_mask)`. The model is trained to maximize the log probability of the correct item. We use multiple masking levels (1, 2, … slots masked) so the policy can handle both “fill one hole” and “build from almost empty” settings.

After SFT, the model can complete outfits in a valid way, but it has no notion of “style” or “looking good”—that comes from the reward model and RL.

### Step 5: Phase 2 — Preference Data and Reward Modeling

- **Preference collection:**  
  - Sample an incomplete outfit (e.g. random empty slots).  
  - Use the SFT policy to **generate two completions** → Outfit A and Outfit B.  
  - **Render** both as composite images (grid of item thumbnails).  
  - Send both images to a **Vision–Language Model** (e.g. GPT, Gemini, Gemma) with a prompt like: “Which of these two outfits (A or B) is more stylistically coherent and looks better? Explain briefly, then output exactly ‘A’ or ‘B’.”  
  - Store `(outfit_a, outfit_b, winner)`.

- **Reward model (RM):** A small network (e.g. embedding layer + MLP) that maps a **full outfit vector** to a scalar score in `[0, 1]`. It is trained so that the **preferred** outfit gets a higher score than the other. 
  
- **Why a separate RM?** Querying the VLM on every RL step would be slow and expensive. Training a small RM on the collected preferences gives a fast, differentiable proxy for “how good does this outfit look?”

### Step 6: Phase 3 — Reinforcement Learning (PPO)

This phase is the core of the project: we treat outfit completion as a **Markov Decision Process (MDP)** and optimize the policy with **Proximal Policy Optimization (PPO)** so that it maximizes the reward model's score while staying close to the SFT policy.

#### What is a Markov Decision Process?

An **MDP** is a standard framework for sequential decision-making. It consists of:

- **State space \($\mathcal{S}$\):** The set of situations the agent can be in. The agent does not observe the full history, only the current state (the "Markov" property: the future depends only on the present).

- **Action space \($\mathcal{A}$\):** The set of choices the agent can make in each state.

- **Transition:** Given state \(s\) and action \(a\), the environment moves to a new state \(s'\). Transitions can be stochastic; here they are deterministic.

- **Reward:** A scalar signal \(r(s, a, s')\) (or \(r(s')\)) that we want to maximize in expectation.

- **Policy \($\pi(a|s)$\):** A mapping from states to a distribution over actions. We train a parameterized policy (the transformer) so that acting according to \($\pi$\) yields high cumulative reward.

**In this project, the MDP is single-step per "episode":**

| MDP concept | In the outfit pipeline |
|-------------|------------------------|
| **State \(s\)** | An **incomplete** outfit vector (at least one slot is `0`). Example: `[0, 142, 0, 89]` — no top, bottom = 142, no shoes, accessory = 89. |
| **Action \(a\)** | One **item ID** chosen for the **first** empty slot (left-to-right). The environment uses `get_first_empty_slot(outfit)` (found in `environment.py`) to decide which slot index to fill; only item IDs that belong to that slot's category are valid (enforced via action masking). |
| **Transition** | Deterministic: the chosen item is written into that slot. From `[0, 142, 0, 89]` and action "item 31 (top)", the new state is `[31, 142, 0, 89]`. |
| **Reward \(r\)** | The **reward model**'s score for the **completed** outfit after this one-step fill.  |

So each "rollout" in training is: sample a batch of incomplete outfits (states) → for each, the policy picks one item for the first empty slot (action) → we get one reward per example from the RM. Building a full outfit from empty is done at **inference** by repeatedly applying this one-step policy (see Step 7).

#### What is PPO (Proximal Policy Optimization)?

**PPO** is a policy-gradient method that updates the policy in a way that avoids too-large steps, which can otherwise destabilize training. Two main ideas:

1. **Importance sampling:** We collect data (state, action, reward) under the **current** policy \($\pi_\theta$\), but we may want to reuse or reweight that data when \($\theta$\) has changed. The **importance ratio** is $\rho = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$. The policy gradient can be estimated with \($\rho \cdot r$\); if \($\rho$\) is large, the old action is much more likely under the new policy and the update can be too aggressive.

2. **Clipping (or penalty) to keep updates "proximal":** PPO limits how much the policy can change in one update. In the clipped objective, the surrogate is $\min\bigl(\rho \cdot A,\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon) \cdot A\bigr)$, so the ratio is not allowed to drift too far from 1. Alternatively (and as in this project), a **KL penalty** is added so that \($\pi_\theta$\) stays close to a **reference policy** \($\pi_{\text{ref}}$\) (here, the frozen SFT policy). That discourages the policy from collapsing to a single high-reward action and preserves diversity.

**In this project, the PPO step (in `src/train_ppo.py`) works as follows:**

1. **Batch of states:** Each PPO step samples a batch of incomplete outfits via `get_random_incomplete_outfit(..., min_empty=1, max_empty=NUM_SLOTS-1)`. For each outfit we get the first empty slot index with `env.get_first_empty_slot(outfit)` and build an **action mask** so that only item IDs for that slot's category are valid (same idea as in SFT).

2. **Rollout:** The **current policy** \($\pi_\theta$\) samples one action (one item ID) per state using `policy.sample(outfit_t, slot_t, mask, temperature=1.0)`, which runs the transformer forward, applies the mask, and samples from the categorical distribution over valid items. We store the action and its **log probability under the current policy** (`old_log_prob`) for the importance ratio.

3. **Transition and reward:** For each sample we form the **completed** outfit by writing the chosen item into the first empty slot. Reward is computed in one batch: `reward = reward_model(comp_t)` — each entry is a scalar in \([0,1]\).

4. **Reference policy and KL:** A **frozen copy** of the SFT policy (`ref_policy`) is kept. We compute the log probability of the **same** actions under the reference policy: `ref_log_prob = ref_policy.get_log_probs(outfit_t, slot_t, action, mask)`. The KL term is approximated as $\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \mathbb{E}[\log \pi_\theta - \log \pi_{\text{ref}}]$. In the code this is `kl = (old_log_prob - ref_log_prob).mean()` (over the batch).

5. **Surrogate and loss:** The importance ratio is `ratio = exp(log_prob - old_log_prob)` where `log_prob` is from the **current** policy (re-evaluated after the graph is set up for backprop). The surrogate is `surr = (ratio * reward).mean()` — we want to maximize this, so the loss includes `-surr`. The full loss is `loss = -surr + kl_coef * kl`. So we **maximize** expected reward (via the surrogate) while **penalizing** deviation from the reference policy. Hyperparameters such as `PPO_KL_COEF` (default 0.1), `PPO_LR`, and `PPO_BATCH_SIZE` are in `config.py`.

6. **Update:** Optimizer step with gradient clipping: `torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)` to avoid exploding gradients.

**Why the KL penalty?** Without it, the policy could collapse to always choosing the single item that gets the highest reward model score, regardless of the rest of the outfit (reward hacking). The KL term ties the policy to the SFT policy, so it keeps diverse, context-aware behavior while improving style as judged by the reward model.

### Step 7: Inference — Building a Full Outfit

- Start from an **empty** outfit `[0, 0, 0, 0]` (or a partial one).  
- **Loop:** Find the first empty slot; get valid item IDs for that slot; run the policy (with action mask) to sample one item; write it into the slot.  
- Repeat until no slot is empty.  
- Optionally **render** the final outfit vector as a grid image using the corpus thumbnails.

This is **autoregressive** in the sense that we fill one slot at a time, each step conditioning on the current (partially filled) outfit.

---

## Project Structure

| Path | Purpose |
|------|--------|
| `src/config.py` | Slots (Top, Bottom, Shoes, Accessory), paths, and hyperparameters (SFT, RM, PPO, VLM). |
| `src/data/` | `Catalog` (load catalog CSV, query by slot/style), synthetic outfit generation, SFT example creation, random incomplete outfits. |
| `src/environment.py` | `OutfitEnvironment`: valid actions per slot, render outfit (or pair A/B) as a single image. |
| `src/models/` | `PolicyNetwork` (transformer policy with action masking), `RewardModel` (outfit vector → scalar). |
| `src/vlm_judge.py` | Call OpenAI (GPT-4o) or Google (Gemini/Gemma) to compare two outfit images and return A/B preference. |
| `src/train_sft.py` | Phase 1: train policy on masked-outfit → target item. |
| `src/collect_preferences.py` | Generate (outfit_a, outfit_b, winner) via SFT + VLM judge; save to `dataset_preferences.json`. |
| `src/train_reward.py` | Train reward model on preference pairs. |
| `src/train_ppo.py` | PPO loop: policy + reference policy + reward model; optimize reward minus KL. |
| `src/inference.py` | Load policy, generate full outfit from empty/partial vector, optionally save rendered image. |
| `scripts/prepare_fpi_corpus.py` | Build corpus from Fashion Product Images (FPI) zip; writes `data/catalog.csv` and extracts images to `data/images/`. |
| `scripts/eval_reward_vs_vlm.py` | Evaluate reward model against VLM preferences. |
| `scripts/render_preference_outfits.py` | Render outfit pairs for inspection. |
| `run_pipeline.py` | Single entry point: prepare data (if needed) → SFT → preferences (or dummy) → RM → PPO → inference. |

Data and outputs:

- **Input:** `data/catalog.csv` (columns e.g. `item_id`, `category`, `image_path`, `style_tag`). Images under `data/images/` (paths in CSV relative to `data/`).
- **Outputs:** `outputs/sft_model.pt`, `outputs/dataset_preferences.json`, `outputs/reward_model.pt`, `outputs/ppo_model.pt`, and generated outfit images.

---

## How to Run

### Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Download Dataset

1. Download the Fashion Product Images (FPI) dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).
2. Rename the file to `fashion_product_images(small).zip` and place the zip file in the `data` folder. (Create the `data` folder if it doesn't exist.)


### Option A: Full pipeline (one script)

From the project root:

```bash
python run_pipeline.py
```

This will:

1. Run a corpus-prep step if the script finds no existing catalog (see Option B step 1 if you need to build the corpus first, e.g. with `scripts/prepare_fpi_corpus.py`).
2. Train SFT (`outputs/sft_model.pt`).
3. Collect preferences via VLM if `OPENAI_API_KEY` or `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) is set; otherwise create **dummy** preferences so the reward model and PPO can still run.
4. Train the reward model.
5. Run PPO.
6. Run inference and save an outfit image to `outputs/outfit.png`.

You can run the full pipeline without any API key: preference collection will be skipped and dummy preferences used for the RM/PPO demo.

### Option B: Step-by-step

**1. Prepare the corpus**

You need `data/catalog.csv` with columns: `item_id`, `category`, `image_path`, `style_tag`. Images should live under `data/images/` (paths in CSV relative to `data/`).

- **Using the Fashion Product Images (FPI) dataset:**  
  Confirm that the dataset zip file (e.g. `fashion_product_images(small).zip`) is in `data/`. Then:

```bash
python scripts/prepare_fpi_corpus.py
```

This extracts images and writes `data/catalog.csv`.

- Or provide your own `data/catalog.csv` (and images) in the same format.

**2. Train SFT**

```bash
python -m src.train_sft --corpus data/catalog.csv --num-outfits 5000 --epochs 20
```

Defaults use `data/catalog.csv` from config if you omit `--corpus`. Model is saved to `outputs/sft_model.pt`.

**3. Collect preferences (optional; requires API key)**

Set an API key, then run:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."
# or
$env:GOOGLE_API_KEY = "..."

python -m src.collect_preferences --sft-model outputs/sft_model.pt --num-pairs 1000 --backend openai
```

Use `--backend gemini` or `gemma` for Google. Preferences are saved to `outputs/dataset_preferences.json`. Without this step, you need dummy preferences (e.g. as generated by `run_pipeline.py`) to train the reward model.

**4. Train the reward model**

```bash
python -m src.train_reward --preferences outputs/dataset_preferences.json
```

Saves `outputs/reward_model.pt`.

**5. Train PPO**

```bash
python -m src.train_ppo --sft-model outputs/sft_model.pt --reward-model outputs/reward_model.pt
```

Saves `outputs/ppo_model.pt`. You can tune `--steps`, `--epochs`, `--kl-coef`, etc.

**6. Generate an outfit**

```bash
python -m src.inference --model outputs/ppo_model.pt --save-image outputs/outfit.png
```

If `outputs/ppo_model.pt` is missing, the script falls back to `outputs/sft_model.pt`. Use `--corpus data/catalog.csv` if your catalog is elsewhere, and `--temperature` to control sampling diversity.

### Resources

- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

