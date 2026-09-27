from pathlib import Path
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "ddos_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"
RESULT_PATH = ROOT / "models" / "evaluation_results.json"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label"])
y = df["Label"].astype(int)

X = X.replace(
    [float("inf"), float("-inf")],
    float("nan")
).fillna(0)

# Chronological test set
split_index = int(len(df) * 0.8)

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

results = {
    "test_samples": int(len(y_test)),
    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),
    "precision": float(
        precision_score(y_test, predictions)
    ),
    "recall": float(
        recall_score(y_test, predictions)
    ),
    "f1_score": float(
        f1_score(y_test, predictions)
    ),
    "confusion_matrix": cm.tolist()
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation results saved!")
print(json.dumps(results, indent=4))