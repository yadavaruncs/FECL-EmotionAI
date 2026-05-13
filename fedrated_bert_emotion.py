# =============================================================================
# FEDERATED BERT — EMOTION DETECTION IN MENTAL HEALTH DIARIES
# FINALIZED LOCAL HUGGING FACE VERSION (NO ANTHROPIC API)
#
# Arun Yadav (23/CS/081)
# Devesh Kadam (23/CS/130)
#
# Features:
# - Federated Learning (FedAvg)
# - Differential Privacy (DP-SGD)
# - BERT-based Emotion Detection
# - Hugging Face LOCAL inference
# - No paid APIs required
# - Fully offline after model download
# =============================================================================

import os
import re
import copy
import json
import math
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torch.utils.data import Dataset, DataLoader, random_split

    from transformers import (
        BertModel,
        BertTokenizer,
        get_linear_schedule_with_warmup,
        pipeline
    )

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        classification_report
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    DL_AVAILABLE = True

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[setup] Using device: {DEVICE}")

except ImportError as e:
    DL_AVAILABLE = False
    print(f"[setup] Missing libraries: {e}")

# =============================================================================
# CONFIG
# =============================================================================

EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "anxiety",
    "depression"
]

NUM_CLASSES = len(EMOTION_LABELS)

BERT_MODEL_NAME = "bert-base-uncased"

HF_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

MAX_SEQ_LEN = 128
HIDDEN_DIM = 768
PROJECTION_DIM = 256
DROPOUT = 0.3

CONTRASTIVE_TEMP = 0.07
CONTRASTIVE_WEIGHT = 0.3

NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2

BATCH_SIZE = 16

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

WARMUP_STEPS = 50

MAX_GRAD_NORM = 1.0

USE_DP = True

DP_EPSILON = 1.0
DP_DELTA = 1e-5
DP_NOISE_MULTIPLIER = 1.2

SEED = 42

OUTPUT_DIR = "outputs"

random.seed(SEED)
np.random.seed(SEED)

if DL_AVAILABLE:
    torch.manual_seed(SEED)

# =============================================================================
# LOCAL HUGGING FACE EMOTION PIPELINE
# =============================================================================

print("\n[HF] Loading local emotion model...")

emotion_classifier = pipeline(
    task="text-classification",
    model=HF_EMOTION_MODEL,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

print("[HF] Emotion model loaded successfully")

# =============================================================================
# LABEL MAPPING
# =============================================================================

LABEL_MAPPING = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "neutral": "depression",
    "disgust": "anger",
    "surprise": "anxiety"
}

# =============================================================================
# SYNTHETIC DATA
# =============================================================================

DIARY_TEMPLATES = {
    "joy": [
        "Today was genuinely good.",
        "I felt happy and grateful.",
        "Spent time laughing with friends.",
        "Everything felt peaceful today."
    ],

    "sadness": [
        "I cried again today.",
        "Everything feels empty.",
        "I miss how things used to be.",
        "Nobody understands me."
    ],

    "anger": [
        "I got extremely frustrated today.",
        "People keep disappointing me.",
        "Everything irritated me.",
        "I snapped at someone."
    ],

    "fear": [
        "I am scared of what might happen.",
        "I could not sleep properly.",
        "Something feels wrong.",
        "I keep imagining worst-case scenarios."
    ],

    "anxiety": [
        "My thoughts will not slow down.",
        "I had a panic attack today.",
        "I feel constantly nervous.",
        "My chest feels tight from stress."
    ],

    "depression": [
        "I could not get out of bed.",
        "Nothing feels meaningful anymore.",
        "I feel emotionally numb.",
        "Everything feels heavy."
    ]
}

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(n_samples=600):

    texts = []
    labels = []

    per_class = n_samples // NUM_CLASSES

    label_map = {
        lbl: i for i, lbl in enumerate(EMOTION_LABELS)
    }

    for emotion, templates in DIARY_TEMPLATES.items():

        label_idx = label_map[emotion]

        for _ in range(per_class):

            text = random.choice(templates)

            texts.append(text)
            labels.append(label_idx)

    combined = list(zip(texts, labels))

    random.shuffle(combined)

    texts, labels = zip(*combined)

    return list(texts), list(labels)

# =============================================================================
# CLEAN TEXT
# =============================================================================

def clean_text(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# =============================================================================
# PREPROCESSOR
# =============================================================================

class TextPreprocessor:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL_NAME
        )

    def encode(self, text):

        text = clean_text(text)

        return self.tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

# =============================================================================
# DATASET
# =============================================================================

class DiaryDataset(Dataset):

    def __init__(self, texts, labels, preprocessor):

        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        enc = self.preprocessor.encode(self.texts[idx])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

# =============================================================================
# DATALOADER
# =============================================================================

def make_loaders(texts, labels, preprocessor):

    dataset = DiaryDataset(
        texts,
        labels,
        preprocessor
    )

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE
    )

    return train_loader, val_loader

# =============================================================================
# MODEL
# =============================================================================

class BERTEmotionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            BERT_MODEL_NAME
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.pooler_output

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits

# =============================================================================
# DIFFERENTIAL PRIVACY
# =============================================================================

class DifferentialPrivacy:

    def __init__(self):

        self.noise_multiplier = DP_NOISE_MULTIPLIER

    def add_noise(self, model):

        with torch.no_grad():

            for p in model.parameters():

                if p.grad is not None:

                    noise = (
                        torch.randn_like(p.grad)
                        * self.noise_multiplier
                    )

                    p.grad += noise

# =============================================================================
# FEDERATED CLIENT
# =============================================================================

class FederatedClient:

    def __init__(
        self,
        client_id,
        texts,
        labels,
        preprocessor
    ):

        self.client_id = client_id

        self.train_loader, self.val_loader = make_loaders(
            texts,
            labels,
            preprocessor
        )

        self.dp = DifferentialPrivacy()

    def train(self, global_weights):

        model = BERTEmotionClassifier().to(DEVICE)

        model.load_state_dict(
            copy.deepcopy(global_weights)
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(LOCAL_EPOCHS):

            for batch in self.train_loader:

                ids = batch["input_ids"].to(DEVICE)

                mask = batch["attention_mask"].to(DEVICE)

                labels = batch["label"].to(DEVICE)

                optimizer.zero_grad()

                logits = model(ids, mask)

                loss = criterion(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )

                if USE_DP:
                    self.dp.add_noise(model)

                optimizer.step()

        return model.state_dict()

# =============================================================================
# FEDERATED SERVER
# =============================================================================

class FederatedServer:

    def __init__(self):

        self.global_model = BERTEmotionClassifier().to(
            DEVICE
        )

    def get_weights(self):

        return copy.deepcopy(
            self.global_model.state_dict()
        )

    def fedavg(self, client_weights):

        avg_weights = copy.deepcopy(client_weights[0])

        for key in avg_weights.keys():

            for i in range(1, len(client_weights)):

                avg_weights[key] += client_weights[i][key]

            avg_weights[key] = torch.div(
                avg_weights[key],
                len(client_weights)
            )

        self.global_model.load_state_dict(avg_weights)

# =============================================================================
# HUGGING FACE LOCAL INFERENCE
# =============================================================================

def analyze_with_local_hf(text):

    print("\n================================================")
    print("LOCAL HUGGING FACE EMOTION ANALYSIS")
    print("================================================")

    print(f"\nInput:\n{text}")

    result = emotion_classifier(text)[0]

    emotions = {}

    for item in result:

        label = item["label"].lower()

        mapped = LABEL_MAPPING.get(label, label)

        emotions[mapped] = round(
            item["score"] * 100,
            2
        )

    primary = max(emotions, key=emotions.get)

    confidence = emotions[primary]

    print(f"\nPrimary Emotion: {primary.upper()}")

    print(f"Confidence: {confidence:.2f}%")

    print("\nEmotion Scores:\n")

    for emotion, score in sorted(
        emotions.items(),
        key=lambda x: -x[1]
    ):

        bar = "█" * int(score / 5)

        print(f"{emotion:<12} {bar:<20} {score}%")

    return {
        "primary_emotion": primary,
        "confidence": confidence,
        "emotions": emotions
    }

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            ids = batch["input_ids"].to(DEVICE)

            mask = batch["attention_mask"].to(DEVICE)

            labels = batch["label"]

            logits = model(ids, mask)

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())

            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)

    f1 = f1_score(
        targets,
        preds,
        average="macro"
    )

    print("\n================================================")
    print("FINAL EVALUATION")
    print("================================================")

    print(f"Accuracy : {acc:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            targets,
            preds,
            target_names=EMOTION_LABELS
        )
    )

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_pipeline():

    print("\n================================================")
    print("FEDERATED BERT TRAINING")
    print("================================================")

    texts, labels = generate_synthetic_data()

    split_size = len(texts) // NUM_CLIENTS

    client_splits = []

    for i in range(NUM_CLIENTS):

        start = i * split_size

        end = (i + 1) * split_size

        client_splits.append((
            texts[start:end],
            labels[start:end]
        ))

    preprocessor = TextPreprocessor()

    server = FederatedServer()

    clients = []

    for i, (t, l) in enumerate(client_splits):

        clients.append(
            FederatedClient(
                i,
                t,
                l,
                preprocessor
            )
        )

    for rnd in range(NUM_ROUNDS):

        print(f"\n========== ROUND {rnd+1} ==========")

        global_weights = server.get_weights()

        client_weights = []

        for client in clients:

            weights = client.train(global_weights)

            client_weights.append(weights)

            print(f"Client {client.client_id+1} trained")

        server.fedavg(client_weights)

        print("FedAvg aggregation completed")

    evaluate(
        server.global_model,
        clients[0].val_loader
    )

    torch.save(
        server.global_model.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "federated_bert_model.pt"
        )
    )

    print("\nModel saved successfully")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TRAINING
    run_pipeline()

    # LOCAL HF INFERENCE DEMO

    sample_text = """
    I could not get out of bed today.
    Everything feels meaningless and exhausting.
    """

    analyze_with_local_hf(sample_text)