from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"
MODEL_PATH = ROOT / "models" / "deep_ddos_model.pt"
SCALER_PATH = ROOT / "models" / "deep_scaler.pkl"
RESULT_PATH = ROOT / "models" / "deep_evaluation_results.json"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label"])
y = df["Label"].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X = X.astype(np.float32)

# ============================================================
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples: ", len(X_test))
print("Features:", X_train.shape[1])


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)


# ============================================================
# PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train_scaled)
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled)
y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# MODEL
# ============================================================

class SentinelMLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = SentinelMLP(X_train.shape[1]).to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

positive = y_train.sum()
negative = len(y_train) - positive

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# MOVE DATA TO DEVICE
# ============================================================

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# ============================================================
# TRAINING
# ============================================================

epochs = 15

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train_tensor)

    loss = criterion(
        outputs,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss.item():.6f}"
    )


# ============================================================
# EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    outputs = model(X_test_tensor)

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities >= 0.5
    ).int()

predictions = (
    predictions.cpu()
    .numpy()
    .flatten()
)

actual = y_test.values


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predictions
)

precision = precision_score(
    actual,
    predictions
)

recall = recall_score(
    actual,
    predictions
)

f1 = f1_score(
    actual,
    predictions
)

cm = confusion_matrix(
    actual,
    predictions
)


print("\nDeep Learning Evaluation")
print("------------------------")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
        "features": list(X.columns),
    },
    MODEL_PATH
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "test_samples": int(len(actual)),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "confusion_matrix": cm.tolist(),

    "model": "PyTorch MLP",

    "split_method": "chronological_80_20",

    "features": int(X.shape[1])
}

with open(
    RESULT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nDeep learning model saved!")
print("Model:", MODEL_PATH)
print("Scaler:", SCALER_PATH)
print("Results:", RESULT_PATH)