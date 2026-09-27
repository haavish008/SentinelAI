
import joblib
import pandas as pd
import requests

# Load the saved model's feature names
bundle = joblib.load("models/ddos_model.pkl")
features = bundle["features"]

# Load one real record from the cleaned dataset
df = pd.read_csv("data/processed/ddos_cleaned.csv")
sample = df.drop(columns=["Label"]).iloc[0]

# Prepare the API request
payload = {
    "features": {
        name: float(sample[name])
        for name in features
    }
}

# Send record to SentinelAI API
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print("Status code:", response.status_code)
print("API response:", response.json())