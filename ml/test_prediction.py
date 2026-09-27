
import joblib
import pandas as pd

# Load the saved model and feature names
bundle = joblib.load("models/ddos_model.pkl")

model = bundle["model"]
features = bundle["features"]

# Load one cleaned network traffic record
df = pd.read_csv("data/processed/ddos_cleaned.csv")

# Select one row, excluding the label
sample = df.drop(columns=["Label"]).iloc[[0]]

# Keep the same feature order used during training
sample = sample[features]

# Predict
prediction = model.predict(sample)[0]

if prediction == 1:
    print("🚨 DDoS traffic detected")
else:
    print("✅ Benign traffic detected")

print("Predicted label:", prediction)