# FECL-EmotionAI 🧠🔒

A federated-learning-based emotion detection system for diary-style text, built with BERT and a local (offline) HuggingFace inference pipeline.

## What This Project Does

Mental health applications often require sensitive personal text to be sent to central servers for analysis — a privacy risk. This project explores keeping training data local and decentralized by simulating federated training of a BERT-based classifier across multiple clients, combined with gradient noise injection during local training.

The system targets 6 emotional states:
joy · sadness · anger · fear · anxiety · depression

## Key Features

- **Federated Learning (FedAvg)** — 3 simulated clients train a `BERTEmotionClassifier` locally; only model weights are shared and averaged, never raw data
- **Gradient noise injection (DP-SGD–style)** — gradients are norm-clipped and Gaussian noise is added before each optimizer step, to reduce the risk of gradient-based data leakage. Note: this repo does not yet track a formal (ε, δ) privacy budget with an accountant — the noise multiplier is fixed, not derived from a target epsilon, so treat this as a DP-*inspired* heuristic rather than a certified guarantee
- **BERT-based Classification** — `bert-base-uncased` fine-tuned for 6-class emotion classification, trained via the federated loop and saved to `outputs/federated_bert_model.pt`
- **Local HuggingFace Inference (separate pipeline)** — the demo at the bottom of the script uses a different, pretrained model (`j-hartmann/emotion-english-distilroberta-base`) for zero-shot inference, fully offline after the initial download. This is **not** the federated-trained model — it's a separate baseline used to produce a quick demo output
- **Contrastive learning scaffolding** — config constants (`PROJECTION_DIM`, `CONTRASTIVE_TEMP`, `CONTRASTIVE_WEIGHT`) are defined for a future contrastive objective, but the contrastive loss itself is not yet implemented in the training loop

## Architecture

```
Diary Text
   │
   ▼
BERT Tokenizer (bert-base-uncased, max_len=128)
   │
   ▼
Federated Training (3 clients)
  Client 1 / Client 2 / Client 3
  → local training, grad clip + noise
   │
   ▼
FedAvg → Global Model → saved to outputs/federated_bert_model.pt
```

Separately, at the end of the script, a **pretrained** HuggingFace pipeline (`j-hartmann/emotion-english-distilroberta-base`) is run on a sample diary entry to print a demo emotion breakdown. Its 7-way output is remapped onto this project's 6 labels as follows — note this mapping is approximate, not a validated clinical mapping:

| HF model label | Mapped to |
|---|---|
| joy | joy |
| sadness | sadness |
| anger | anger |
| fear | fear |
| disgust | anger |
| neutral | depression |
| surprise | anxiety |

Because `neutral` is mapped to `depression`, neutral-sounding text can surface as "depression" in the demo output — worth keeping in mind when interpreting results.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Deep Learning | PyTorch |
| NLP Model | BERT (HuggingFace Transformers) |
| Federated Learning | Custom FedAvg implementation |
| Privacy | Gradient clipping + Gaussian noise (DP-SGD–style, no formal accountant) |
| Metrics | Scikit-learn |

## Training Configuration

| Parameter | Value |
|---|---|
| Federation Rounds | 5 |
| Clients | 3 |
| Local Epochs | 2 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Gradient Clip Norm | 1.0 |
| Noise Multiplier | 1.2 |

## Data

Training uses **synthetic** diary text generated from a small set of hand-written template sentences (4 per emotion class), sampled with replacement to build the dataset. This is useful for exercising the pipeline end-to-end, but the resulting train/val split will contain many near-duplicate sentences — accuracy/F1 numbers from this data shouldn't be read as representative of real-world diary text performance. Swapping in a real labeled dataset is a natural next step.

## How to Run

**1. Install dependencies**

```bash
pip install torch transformers scikit-learn matplotlib seaborn
```

**2. Run the full pipeline**

```bash
python fedrated_bert_emotion.py
```

This will:
- Generate synthetic diary data
- Train the federated BERT classifier across 3 clients for 5 rounds
- Evaluate on a validation split (accuracy + macro F1 + classification report)
- Save the federated model to `outputs/federated_bert_model.pt`
- Separately, run the pretrained HuggingFace pipeline on a sample diary entry and print an emotion breakdown (see the mapping caveat above)

## Known Limitations / Next Steps

- Contrastive loss is configured but not implemented — either wire it into training or drop the config constants
- No formal DP accountant — epsilon/delta aren't currently derived from the noise multiplier
- Federated-trained model and the demo inference model are not connected; consider loading `federated_bert_model.pt` for the demo instead of the separate pretrained pipeline
- Synthetic dataset is small and repetitive; a real diary/emotion dataset would give more meaningful metrics

## Author

Arun Yadav — B.Tech CSE, Delhi Technological University
