# Temporary script to update README section 6 with correct encoding
import pathlib

path = pathlib.Path(r"C:\Users\ellio\Desktop\styleRLHF\README.md")
text = path.read_text(encoding="utf-8")

# Use exact Unicode from file: curly apostrophe U+2019
old_block = """### Step 6: Phase 3 — Reinforcement Learning (PPO)

- **MDP:**  
  - **State:** An incomplete outfit vector (at least one slot is 0).  
  - **Action:** One item ID for the **first** empty slot (left-to-right).  
  - **Transition:** The chosen item is written into that slot.  
  - **Reward:** The **reward model**\u2019s score for the **completed** outfit (after this one-step fill).

- **PPO (simplified, single-step):**  
  - **Rollout:** For a batch of states (incomplete outfits), the current **policy** samples one action (one item) for the first empty slot.  
  - **Reward:** `r = RM(completed_outfit)`.  
  - **KL penalty:** We keep a **frozen copy** of the SFT policy (reference policy). We want the current policy\u2019s distribution over actions to stay close to the reference. So we add a penalty `β * KL(π_current || π_ref)`.  
  - **Update:** Maximize `E[reward - β * KL]` (in practice: policy gradient with clipped objective and KL term).

- **Why KL penalty?** Without it, the policy can "collapse" to always picking the single item that gets the highest RM score, ignoring context. The KL term encourages the policy to stay close to the SFT policy, preserving diversity and sensible behavior while improving style."""

new_block = r"""### Step 6: Phase 3 — Reinforcement Learning (PPO)

This phase is the core of the project: we treat outfit completion as a **Markov Decision Process (MDP)** and optimize the policy with **Proximal Policy Optimization (PPO)** so that it maximizes the reward model's score while staying close to the SFT policy.

#### What is a Markov Decision Process?

An **MDP** is a standard framework for sequential decision-making. It consists of:

- **State space \(\mathcal{S}\):** The set of situations the agent can be in. The agent does not observe the full history, only the current state (the "Markov" property: the future depends only on the present).

- **Action space \(\mathcal{A}\):** The set of choices the agent can make in each state.

- **Transition:** Given state \(s\) and action \(a\), the environment moves to a new state \(s'\). Transitions can be stochastic; here they are deterministic.

- **Reward:** A scalar signal \(r(s, a, s')\) (or \(r(s')\)) that we want to maximize in expectation.

- **Policy \(\pi(a|s)\):** A mapping from states to a distribution over actions. We train a parameterized policy (the transformer) so that acting according to \(\pi\) yields high cumulative reward.

**In this project, the MDP is single-step per "episode":**

| MDP concept | In the outfit pipeline |
|-------------|------------------------|
| **State \(s\)** | An **incomplete** outfit vector (at least one slot is `0`). Example: `[0, 142, 0, 89]` — no top, bottom = 142, no shoes, accessory = 89. |
| **Action \(a\)** | One **item ID** chosen for the **first** empty slot (left-to-right). The environment uses `get_first_empty_slot(outfit)` (e.g. in `environment.py`) to decide which slot index to fill; only item IDs that belong to that slot's category are valid (enforced via action masking). |
| **Transition** | Deterministic: the chosen item is written into that slot. From `[0, 142, 0, 89]` and action "item 31 (top)", the new state is `[31, 142, 0, 89]`. |
| **Reward \(r\)** | The **reward model**'s score for the **completed** outfit after this one-step fill. So \(r = \text{RM}(\text{completed\_outfit})\). No reward is given for intermediate steps; we only score the outfit once the single missing item is filled. |

So each "rollout" in training is: sample a batch of incomplete outfits (states) → for each, the policy picks one item for the first empty slot (action) → we get one reward per example from the RM. Building a full outfit from empty is done at **inference** by repeatedly applying this one-step policy (see Step 7).

#### What is PPO (Proximal Policy Optimization)?

**PPO** is a policy-gradient method that updates the policy in a way that avoids too-large steps, which can otherwise destabilize training. Two main ideas:

1. **Importance sampling:** We collect data (state, action, reward) under the **current** policy \(\pi_\theta\), but we may want to reuse or reweight that data when \(\theta\) has changed. The **importance ratio** is \(\rho = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}\). The policy gradient can be estimated with \(\rho \cdot r\); if \(\rho\) is large, the old action is much more likely under the new policy and the update can be too aggressive.

2. **Clipping (or penalty) to keep updates "proximal":** PPO limits how much the policy can change in one update. In the clipped objective, the surrogate is \(\min\bigl(\rho \cdot A,\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon) \cdot A\bigr)\), so the ratio is not allowed to drift too far from 1. Alternatively (and as in this project), a **KL penalty** is added so that \(\pi_\theta\) stays close to a **reference policy** \(\pi_{\text{ref}}\) (here, the frozen SFT policy). That discourages the policy from collapsing to a single high-reward action and preserves diversity.

**In this project, the PPO step (in `src/train_ppo.py`) works as follows:**

1. **Batch of states:** Each PPO step samples a batch of incomplete outfits via `get_random_incomplete_outfit(..., min_empty=1, max_empty=NUM_SLOTS-1)`. For each outfit we get the first empty slot index with `env.get_first_empty_slot(outfit)` and build an **action mask** so that only item IDs for that slot's category are valid (same idea as in SFT).

2. **Rollout:** The **current policy** \(\pi_\theta\) samples one action (one item ID) per state using `policy.sample(outfit_t, slot_t, mask, temperature=1.0)`, which runs the transformer forward, applies the mask, and samples from the categorical distribution over valid items. We store the action and its **log probability under the current policy** (`old_log_prob`) for the importance ratio.

3. **Transition and reward:** For each sample we form the **completed** outfit by writing the chosen item into the first empty slot. Reward is computed in one batch: `reward = reward_model(comp_t)` — each entry is a scalar in \([0,1]\).

4. **Reference policy and KL:** A **frozen copy** of the SFT policy (`ref_policy`) is kept. We compute the log probability of the **same** actions under the reference policy: `ref_log_prob = ref_policy.get_log_probs(outfit_t, slot_t, action, mask)`. The KL term is approximated as \(\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \mathbb{E}[\log \pi_\theta - \log \pi_{\text{ref}}]\); in the code this is `kl = (old_log_prob - ref_log_prob).mean()` (over the batch).

5. **Surrogate and loss:** The importance ratio is `ratio = exp(log_prob - old_log_prob)` where `log_prob` is from the **current** policy (re-evaluated after the graph is set up for backprop). The surrogate is `surr = (ratio * reward).mean()` — we want to maximize this, so the loss includes `-surr`. The full loss is `loss = -surr + kl_coef * kl`. So we **maximize** expected reward (via the surrogate) while **penalizing** deviation from the reference policy. Hyperparameters such as `PPO_KL_COEF` (default 0.1), `PPO_LR`, and `PPO_BATCH_SIZE` are in `config.py`.

6. **Update:** Optimizer step with gradient clipping: `torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)` to avoid exploding gradients.

**Why the KL penalty?** Without it, the policy could collapse to always choosing the single item that gets the highest reward model score, regardless of the rest of the outfit (reward hacking). The KL term ties the policy to the SFT policy, so it keeps diverse, context-aware behavior while improving style as judged by the reward model."""

if old_block not in text:
    raise SystemExit("Old block not found in file")
text = text.replace(old_block, new_block, 1)

path.write_text(text, encoding="utf-8")
print("README section 6 updated successfully.")
