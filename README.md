FECL-EmotionAI 🧠🔒

A privacy-preserving emotion detection system for mental health diary text, built using Federated Learning, Contrastive Learning, and BERT.


What This Project Does

Mental health applications often require sensitive personal text to be sent to central servers for analysis — a significant privacy risk. This project solves that by keeping user data local and decentralized while still training a powerful emotion detection model collaboratively.

The system detects 6 emotional states from diary-style text:
joy · sadness · anger · fear · anxiety · depression


Key Features


Federated Learning (FedAvg) — 3 simulated clients train locally; only model weights are shared, never raw data
Differential Privacy (DP-SGD) — Gaussian noise (ε=1.0, δ=1e-5) added during training to prevent gradient leakage
BERT-based Classification — bert-base-uncased fine-tuned for emotion classification on mental health diary text
Contrastive Learning — Improves embedding quality with temperature-scaled contrastive loss (τ=0.07)
Local HuggingFace Inference — Uses j-hartmann/emotion-english-distilroberta-base for zero-shot inference; fully offline after download



Architecture

Mental Health Diary Text
        │
        ▼
┌─────────────────┐
│  BERT Tokenizer │  (bert-base-uncased, max_len=128)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Federated Training (3 Clients)       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Client 1 │ │ Client 2 │ │ Client 3 │    │
│  │ Local    │ │ Local    │ │ Local    │    │
│  │ Data     │ │ Data     │ │ Data     │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │DP-SGD      │DP-SGD      │DP-SGD     │
│       └────────────┼────────────┘           │
│                    │ FedAvg                  │
│            ┌───────┴────────┐               │
│            │  Global Model  │               │
│            └────────────────┘               │
└─────────────────────────────────────────────┘
         │
         ▼
  Emotion Classification
  (6 classes + confidence scores)


Tech Stack

ComponentTechnologyLanguagePython 3.9+Deep LearningPyTorchNLP ModelBERT (HuggingFace Transformers)Federated LearningCustom FedAvg implementationPrivacyDP-SGD with Gaussian noiseMetricsScikit-learnVisualizationMatplotlib, Seaborn


Training Configuration

ParameterValueFederation Rounds5Clients3Local Epochs2Batch Size16Learning Rate2e-5DP Noise Multiplier1.2DP Epsilon (ε)1.0


How to Run

1. Install dependencies

bashpip install torch transformers scikit-learn matplotlib seaborn

2. Run the full pipeline

bashpython fedrated_bert_emotion.py

This will:


Generate synthetic mental health diary data
Train across 3 federated clients for 5 rounds with DP-SGD
Evaluate on validation set (accuracy + macro F1 + full classification report)
Run a local HuggingFace inference demo on a sample diary entry
Save the trained model to outputs/federated_bert_model.pt


3. Sample output

Primary Emotion: DEPRESSION
Confidence: 87.43%

Emotion Scores:
depression   ████████████████     87.43%
sadness      ███                  12.10%
anxiety      █                    0.47%


Privacy Guarantees


Raw text never leaves the local client — only model weight updates are shared
DP-SGD adds calibrated Gaussian noise at each gradient step
Formal privacy guarantee: (ε=1.0, δ=1e-5)-differential privacy



Author

Arun Yadav — B.Tech CSE, Delhi Technological University
