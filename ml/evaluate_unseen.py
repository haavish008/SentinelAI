from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "ddos_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"

print("Loading trained model...")
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Clean data
X = df.drop(columns=["Label"])

X = X.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

X = X.fillna(0)

y = df["Label"].astype(int)

# --------------------------------------------------
# Chronological split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("\nUnseen chronological test set")
print("Test rows:", len(X_test))

# Prediction
predictions = model.predict(X_test)

# Results
print("\n--- Unseen Traffic Results ---")

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)