# FECL-EmotionAI — Federated Emotion Classification with Privacy

Built by **Arun Yadav**.

A compact experiment measuring how much emotion-classification
performance actually costs when you move from centralized training to
**federated learning**, and then add **differential privacy** on top.

![Results](results/results.png)

---

## 1. The problem

Text emotion classifiers are trained on people's raw messages, journal entries,
support tickets, chat logs. Centralized training means pooling all of that raw
text on one server — a real privacy liability (and, in regulated domains, a
compliance problem) even before anything goes wrong. The question this project
answers with real numbers, not assumptions: **if we stop centralizing raw data,
how much accuracy do we actually lose — and how much more do we lose if we also
add a formal privacy guarantee?**

## 2. The solution

Train a shared emotion classifier using **federated learning** (FedAvg): each
simulated client trains locally on its own (non-IID) slice of data and only
ever sends model **weights** back to a server, never raw text. Optionally wrap
local training in **DP-SGD** (via Opacus) so each client's contribution comes
with an actual, computed `(epsilon, delta)` privacy guarantee. Compare all
three regimes — centralized, federated, federated+DP — on one shared held-out
test set.

**Stack:** Python 3.10–3.12, PyTorch, Hugging Face Transformers/Datasets, Opacus, scikit-learn, NumPy, Matplotlib, Streamlit, pytest.

## Pipeline and operations

- **Dataset and feature cache:** `data.py` downloads the official train/validation/test splits from Hugging Face at runtime. Since DistilBERT is frozen, each split is encoded once in batches (default 32) and cached as an `.npz` under `.cache/`. The cache key includes the dataset name, encoder, max sequence length, split, and the full input text sequence, so changed text/configuration gets a different file. A temporary file is written and then atomically replaced into place; labels are taken from the currently loaded dataset when a cache hit is used. Dataset/model downloads and the cache live outside version control.
- **Experiment orchestration:** `run_experiments.py` loads/reuses those embeddings, then runs the centralized baseline, sequential simulated FedAvg clients, the DP noise sweep, the SupCon ablation, an IID-split control, and a clipping-only control. Client partitions are made from the training labels with a seeded Dirichlet draw; empty clients are skipped. Each round trains clients locally and aggregates model weights by sample count. The validation split is logged per round for monitoring; the common global test split is used for reported comparison metrics.
- **Privacy accounting and constraints:** DP training wraps each client with an Opacus `PrivacyEngine` that is retained across that client's rounds, so its reported epsilon accumulates across participation. The reported sweep value is the maximum client epsilon. DP mode disables SupCon and uses cross-entropy only because SupCon couples examples within a batch, violating DP-SGD's per-example-loss assumption. This is a single-machine simulation with sequential client execution, not a distributed network deployment.
- **Artifacts and recovery:** `train.py` writes a mode checkpoint to `checkpoints/<mode>.pt`; the full experiment script writes checkpoints for the main centralized, federated, and headline DP models plus `results/results.csv` and diagnostic plots. `evaluate.py` reloads embeddings and a checkpoint to evaluate on the held-out test split. The Streamlit app expects `checkpoints/federated.pt` and explicitly refuses to substitute an unrelated pretrained classifier when it is absent. The experiment script stops on an exception and writes the result table only after the experiment stages complete; rerun the failed command after resolving the cause. Checkpoints and embedding caches are local generated state, not committed source artifacts.
- **Testing:** `pytest` covers split loading, embedding-cache correctness, partition coverage, FedAvg arithmetic, empty-client handling, DP epsilon accumulation, frozen-encoder properties, and SupCon edge cases. The data tests intentionally use the real Hugging Face dataset and DistilBERT encoder, so the first test run needs network access/model downloads unless those assets are already cached; only the cache-key checks use a temporary directory.

## 3. Architecture

```
                    ┌─────────────────────────────────────────┐
                    │        dair-ai/emotion (HF dataset)      │
                    │   train / validation / test splits       │
                    └───────────────────┬───────────────────────┘
                                         │
                          Dirichlet(alpha) non-IID split
                                         │
              ┌──────────────┬──────────┴──────────┬──────────────┐
              ▼              ▼                     ▼              
        ┌───────────┐  ┌───────────┐         ┌───────────┐
        │ Client 1  │  │ Client 2  │   ...   │ Client N  │
        │ (local    │  │ (local    │         │ (local    │
        │  emotion  │  │  emotion  │         │  emotion  │
        │  mix)     │  │  mix)     │         │  mix)     │
        └─────┬─────┘  └─────┬─────┘         └─────┬─────┘
              │  frozen DistilBERT (shared, never trained)
              │  -> mean-pooled sentence embedding
              ▼              ▼                     ▼
        ┌───────────┐  ┌───────────┐         ┌───────────┐
        │ trainable │  │ trainable │         │ trainable │
        │  head     │  │  head     │   ...   │  head     │
        │ (+DP-SGD  │  │ (+DP-SGD  │         │ (+DP-SGD  │
        │  optional)│  │  optional)│         │  optional)│
        └─────┬─────┘  └─────┬─────┘         └─────┬─────┘
              │  local weights + n_samples          │
              └──────────────┬───────────────────────┘
                              ▼
                 sample-weighted FedAvg on server
                              │
                              ▼
                     global classification head
                              │
                              ▼
              held-out GLOBAL test set (never seen by any client)
                              │
                              ▼
                Accuracy + Macro-F1 (same code, all approaches)
```

## 4. Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — real
English text, 6 emotion classes (`sadness`, `joy`, `love`, `anger`, `fear`,
`surprise`), official train/validation/test splits (16,000 / 2,000 / 2,000
examples). Downloaded automatically at runtime via the HuggingFace `datasets`
library — no data is bundled in this repo. The class distribution is
imbalanced (`joy` and `sadness` dominate, `surprise` is rare), which is the
reason Macro-F1 is the primary metric (see §9).

## 5. Model, and why the encoder is frozen

**DistilBERT** (`distilbert-base-uncased`, ~66M parameters) turns each
sentence into a 768-dim embedding (attention-mask-aware mean pooling over the
last hidden state). Its parameters are **entirely frozen**
(`requires_grad=False` on every one, asserted directly in
`tests/test_model.py`) — only a small trainable head (~116K parameters: a
shared trunk plus a classifier and a projection layer for SupCon) is trained.

This is a deliberate engineering trade-off, not a corner cut:

- Fine-tuning 66M parameters inside a loop that already multiplies cost by
  `num_clients x num_rounds` is impractical on a laptop/Colab.
- **DP-SGD noise scales with the number of trainable parameters.** A
  66M-parameter fine-tune under DP-SGD would need noise so large the model
  would barely learn — a small head is what keeps the privacy-utility
  trade-off actually *visible* instead of collapsing to noise immediately.
- Since the encoder never changes, its output for a given sentence never
  changes either — so every sentence is encoded **once** and the resulting
  embedding is cached (`data.get_frozen_embeddings`). This is mathematically
  identical to re-running the frozen encoder every step, just without the
  wasted repeated compute, and it's what makes running a full noise-multiplier
  sweep practical on CPU.

## 6. Non-IID client partitioning

`federated.dirichlet_partition` splits the training set across
`config.num_clients` (3 by default) clients using a per-class
`Dirichlet(alpha)` draw (`config.dirichlet_alpha`, default `0.3`): for each
emotion class, the proportion of that class's examples going to each client is
drawn from a Dirichlet distribution. Low alpha → each client ends up with a
skewed, realistic mix of emotions (e.g. mostly `anger` and `fear`); high alpha
→ close to a uniform, IID split. Every example is assigned to exactly one
client — the partition covers the whole training set with no overlap.

## 7. FedAvg

Standard sample-weighted federated averaging (`federated.fedavg_aggregate`),
whiteboard formula:

```
w_global <- Σ_k ( n_k / n_total ) * w_k        where n_total = Σ_k n_k
```

Each client trains locally for `config.local_epochs` epochs, then returns its
updated weights `w_k` and its local sample count `n_k`. Only floating-point
tensors are averaged as weighted sums; any non-floating-point tensor (e.g. an
integer counter/buffer) is copied through instead of incorrectly averaged.
Verified against a hand-computed value in `tests/test_federated.py`.

## 8. Supervised contrastive learning (SupCon)

The trainable head (`model.ClassifierHead`) has a shared trunk feeding two
branches: a classifier and a small projection head, trained jointly with
`CrossEntropy + lambda * SupCon` (`model.supcon_loss`, Khosla et al. 2020,
simplified to one view per sample). The trunk is shared deliberately — if the
classifier and projection heads instead read the frozen embedding
independently (no shared parameters), SupCon's gradient could never influence
the classifier's predictions at all, making it a no-op instead of a real
joint objective. Positives for a given anchor are other same-label samples in
the same batch; a batch with no positive pairs at all safely returns a `0.0`
loss instead of `NaN` (`tests/test_model.py::test_supcon_loss_handles_no_positive_pairs`).
A real ablation (CE-only vs. CE+SupCon, identical partition/seed/rounds) is in
`results/results.csv` under `experiment_group=supcon_ablation` — see §10 for
the honest result and why it looks the way it does.

**SupCon is switched off in every DP run.** DP-SGD's guarantee assumes the
loss is a sum of per-example losses, so each example's clipped gradient
depends on that example alone. SupCon breaks this: each sample's contrastive
term uses every other sample in the batch, so changing one sentence changes
the others' gradients too (the same reason Opacus rejects BatchNorm — but it
cannot detect a batch-coupled loss). `federated._train_local_model` therefore
trains DP clients with cross-entropy only, regardless of `config.use_supcon`.

## 9. Evaluation methodology

Every approach (centralized / federated / federated+DP) is evaluated with the
**same code, on the same held-out global test set** — never on a client's own
local (skewed) training data. Metrics: **accuracy** and **Macro-F1**.
Macro-F1 is the headline metric because the dataset is imbalanced (§4): a
model that mostly predicts `joy` can post deceptively high accuracy while
ignoring rare classes like `surprise`; Macro-F1 weights every class equally
and exposes that failure mode.

## 10. Experiments and results

Default configuration (all in `config.py`): 3 clients, Dirichlet
`alpha=0.3`, 5 federated rounds, 2 local epochs/round, batch size 32, Adam
`lr=1e-3`, `SupCon lambda=0.1, temperature=0.07`, seed 42, DP
`max_grad_norm=1.0, delta=1e-5`, noise-multiplier sweep `[0.5, 1.0, 1.5]`.

All numbers below come from an actual run of `run_experiments.py` in this
environment (CPU only, no GPU), single seed (42) — see `results/results.csv`
for the raw table. Privacy budgets are computed by Opacus's default **PRV
(Privacy Random Variable) accountant**, which gives a tighter epsilon than the
older RDP accountant for the same training (at noise 1.0 the worst client is
epsilon≈1.37 with PRV; RDP would report ≈1.61). The reported epsilon is the
worst (largest) of the three clients' cumulative budgets.

### Headline comparison

![Results](results/results.png)

| Approach | Accuracy | Macro-F1 | epsilon (delta=1e-5) |
|---|---|---|---|
| Centralized | 0.6590 | 0.5606 | — |
| Federated (FedAvg) | 0.5865 | 0.4309 | — |
| Federated + DP (noise=1.0) | 0.4920 | 0.2023 | 1.374 |

Going centralized → federated costs about 13 points of Macro-F1 (0.561 →
0.431) under this non-IID split (`alpha=0.3`, 3 clients). That gap is
**mostly label skew, not federation itself**: the IID-federated control
below (same code, `alpha=100`) reaches 0.549, only ~1 point under
centralized. The partition at `alpha=0.3` is extreme — client 0 has zero
`joy` and zero `love`, client 2 is ~77% `joy`. Adding DP-SGD on top costs
another ~23 points of Macro-F1 at epsilon≈1.37 — and, as the next section
shows, that DP number reflects a collapsed model, not a usable one.

### Privacy-utility trade-off

![Privacy-utility trade-off](results/privacy_utility.png)

| Noise multiplier | epsilon | Accuracy | Macro-F1 |
|---|---|---|---|
| 0 (clipping only, control) | ∞ (no guarantee) | 0.6055 | 0.4268 |
| 0.5 | 10.969 | 0.5085 | 0.2089 |
| 1.0 | 1.374 | 0.4920 | 0.2023 |
| 1.5 | 0.705 | 0.4930 | 0.2023 |

**At this budget (5 rounds x 2 local epochs, batch 32), every DP run with
noise collapses to the two majority classes.** The noise-1.0 model predicts
only `sadness` (1,263 test sentences) and `joy` (737); `love`, `anger`,
`fear` and `surprise` are never predicted. That is why accuracy stays near
0.50 (those two classes are ~64% of the test set) while Macro-F1 falls to
~0.20 — and why the curve is flat: all three noisy runs land in the same
collapsed state, so the sweep does not really trace a trade-off.

The clipping-only control (same DP-SGD machinery, `noise_multiplier=0`)
reaches Macro-F1 0.427, essentially the non-DP federated 0.431. So
**per-sample clipping is not what breaks the model — the added noise is.**
The honest conclusion is that this DP configuration is not usable yet. The
standard fixes are larger batches (noise is added once per batch, so a
bigger batch raises the signal-to-noise ratio of each step) and more
training steps; both are future work (§17).

### SupCon ablation

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| Federated, CE only | 0.5885 | 0.4215 |
| Federated, CE + SupCon | 0.5865 | 0.4309 |

SupCon gives a small Macro-F1 difference (+0.0094) at essentially flat
accuracy (-0.002), using an identical partition/seed/rounds for both runs.
This is **one run per configuration on one seed**, and a 0.009 Macro-F1
difference is within normal seed-to-seed variation — so it is a small,
not-yet-significant improvement. Claiming a real effect would need 3–5 seeds
and a confidence interval. The direction is at least consistent with the
mechanism (SupCon shapes the shared trunk to separate classes, which would
help minority-class F1 more than accuracy), and with a frozen encoder a large
gain would be surprising: SupCon reshapes a ~100K-parameter trunk, not the
encoder itself.

### Controls

| Control | Accuracy | Macro-F1 | What it isolates |
|---|---|---|---|
| Federated, IID (`alpha=100`) | 0.6495 | 0.5494 | Cost of federation alone (vs. centralized 0.5606) |
| Federated + DP, clipping only (`noise=0`) | 0.6055 | 0.4268 | Cost of per-sample clipping alone (vs. federated 0.4309) |

### Confusion matrices

| Centralized | Federated | Federated + DP |
|---|---|---|
| ![Centralized confusion matrix](results/confusion_matrix_centralized.png) | ![Federated confusion matrix](results/confusion_matrix_federated.png) | ![Federated+DP confusion matrix](results/confusion_matrix_federated_dp.png) |

## 11. Engineering decisions

- **Frozen encoder + tiny trainable head.** See §5 — makes federated rounds x
  clients x DP noise levels tractable on a CPU, and keeps DP-SGD noise from
  overwhelming a huge parameter count.
- **Precomputed, cached embeddings.** Since the encoder is frozen, its output
  per sentence is a pure function of the input — computed once, cached to
  `.cache/`, and reused across every experiment. Not a shortcut: mathematically
  identical to re-running the encoder every step.
- **Only 3 clients by default.** Enough to show a real, measurable non-IID
  effect without needing a cluster; `config.num_clients` is one line to raise.
- **Dirichlet non-IID partitioning**, not a random/IID split — a random split
  would hide exactly the failure mode federated learning has to survive in
  practice (see §6, §9). The IID run is kept only as a control (§10).
- **Sample-weighted FedAvg**, not unweighted averaging — a client with more
  local data should influence the global model proportionally more; verified
  against a hand-computed value in tests, not just "trust the library".
- **One shared global test set for every approach.** Reporting client-local
  performance would make federated/DP look artificially good or bad depending
  on which client's data happens to be evaluated.
- **Persistent per-client `PrivacyEngine` across rounds** (`federated.py`) —
  a client's cumulative privacy spend must accumulate over every round it
  participates in, not reset every round; the reported epsilon is the true
  cumulative value from Opacus's PRV accountant (worst client).
- **Cross-entropy only under DP.** SupCon couples samples within a batch,
  which would void DP-SGD's per-example guarantee (§8), so it is disabled in
  DP mode in code, not just by convention.
- **Centralized training budget = 10 epochs** (`num_rounds x local_epochs`),
  so every sample is seen 10 times in both centralized and federated training.
- **Lightweight, fixed experiment design** (5 rounds, 2 local epochs, one
  noise sweep) instead of a broad hyperparameter search — this project is
  built to be explained end-to-end in an interview, not to chase leaderboard
  numbers.

## 12. Installation

Requires Python 3.10–3.12 (`numpy<2.0` is pinned, and NumPy 1.26 has no
wheels for Python 3.13). From inside this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## 13. Usage

All commands are run from the project root.

```bash
# Run the full experiment suite (centralized, federated, DP sweep, SupCon
# ablation) and regenerate results/results.csv, results/results.png, etc.
python run_experiments.py

# Train just one regime and save its checkpoint to checkpoints/<mode>.pt
python train.py --mode centralized
python train.py --mode federated
python train.py --mode federated_dp

# Evaluate a saved checkpoint on the held-out test set
python evaluate.py --checkpoint checkpoints/federated.pt

# Predict the emotion of a single sentence
python predict.py "I can't believe how happy this makes me feel"

# Launch the interactive demo (uses checkpoints/federated.pt)
streamlit run app.py

# Run the test suite (12 tests)
pytest
```

## 14. Project structure

```
FECL-EmotionAI/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.py              # single source of truth for every experiment setting
├── data.py                 # dataset loading + frozen-encoder embedding cache
├── model.py                 # FrozenEncoder, ClassifierHead, SupCon loss
├── federated.py              # Dirichlet partitioning, FedAvg, federated loop
├── privacy.py                 # Opacus DP-SGD wiring, cumulative epsilon accounting
├── train.py                    # centralized training + CLI entry point
├── evaluate.py                  # accuracy / macro-F1 / confusion matrix
├── predict.py                    # single-sentence inference
├── run_experiments.py             # runs everything, writes results/
├── app.py                          # Streamlit demo
├── tests/                          # 12 pytest tests
│   ├── test_data.py               # 3: splits, embedding cache (temp dir), cache key
│   ├── test_federated.py          # 5: partition, FedAvg, empty clients, DP smoke test
│   └── test_model.py              # 4: frozen encoder, head shapes, SupCon edge cases
└── results/
    ├── results.csv
    ├── results.png
    ├── privacy_utility.png
    └── confusion_matrix_*.png
```

## 15. Technologies

Python, PyTorch, HuggingFace Transformers & Datasets, Opacus (DP-SGD),
scikit-learn (metrics), NumPy, Matplotlib, Streamlit, pytest.

## 16. Limitations

- Federated learning is **simulated on one machine** (sequential client
  training) — no real network communication, latency, or client dropout.
- No robust aggregation against malicious/corrupted clients (plain FedAvg
  trusts every client update proportionally to its sample count).
- DP accounting is per-client and independent; no cross-client/cross-round
  composition analysis beyond what Opacus's accountant reports per client.
- Fixed, lightweight hyperparameters by design (§11) — not a hyperparameter
  search.
- **Single seed.** Every number is one run with seed 42; small differences
  (e.g. the SupCon ablation) are not statistically established.
- **The DP configuration collapses** to the two majority classes (§10); the
  DP rows show the formal epsilon accounting working, not a usable private
  model.
- `config.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid an OpenMP
  double-initialization crash on some Windows setups. Intel documents this as
  an unsafe workaround that can give wrong results; fine for a local demo, not
  for production.

## 17. Future work

- **Make DP usable:** larger batches (e.g. 256+) and more rounds, re-running
  the sweep to get a real trade-off curve instead of a collapsed model.
- **Multiple seeds** (3–5) with confidence intervals for every comparison.
- **FedProx** — a proximal term that penalizes local models drifting too far
  from the global model, specifically helpful under strong non-IID data.
- **Secure aggregation** — server only ever sees the *sum* of client updates,
  never any individual client's update in the clear.
- **Communication compression** (quantization/sparsification) for real
  bandwidth-constrained edge clients.
- Real cross-device orchestration (e.g. Flower) instead of a simulated
  sequential loop.

## 18. License

[MIT](LICENSE) — Copyright (c) 2026 Arun Yadav.
