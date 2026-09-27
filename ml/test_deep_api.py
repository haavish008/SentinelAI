from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Take first record
row = df.iloc[0]

# Remove label
features = {
    column: float(row[column])
    for column in df.columns
    if column != "Label"
}


# Send to deep-learning API
response = requests.post(
    "http://127.0.0.1:8000/predict/deep",
    json={"features": features},
    timeout=10
)


print("Status code:", response.status_code)
print("Response:")
print(response.json())

print("\nActual label:")

if row["Label"] == 1:
    print("DDoS")
else:
    print("BENIGN")