from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"
MODEL_PATH = ROOT / "models" / "ddos_model.pkl"
RESULT_PATH = ROOT / "models" / "evaluation_results.json"


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label"])
y = df["Label"].astype(int)

# Clean numeric values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(0)

# -----------------------------
# 2. Chronological 80/20 split
# -----------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# -----------------------------
# 3. Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 5. Save model
# -----------------------------
model_bundle = {
    "model": model,
    "features": list(X.columns),
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_bundle, MODEL_PATH)

# -----------------------------
# 6. Save evaluation results
# -----------------------------
results = {
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "split_method": "chronological_80_20",
}

import json

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nModel saved!")
print(f"Model: {MODEL_PATH}")
print(f"Results: {RESULT_PATH}")