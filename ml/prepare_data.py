
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Find the CSV
files = list(RAW.glob("*DDos*.csv"))

if not files:
    raise FileNotFoundError("DDoS CSV not found in data/raw")

print("Reading:", files[0])
df = pd.read_csv(files[0], low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Remove invalid values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Convert labels to binary:
# BENIGN = 0, DDoS = 1
df["Label"] = (
    df["Label"].astype(str).str.strip().str.upper()
    .map({"BENIGN": 0, "DDOS": 1})
)

# Keep only recognized labels
df.dropna(subset=["Label"], inplace=True)
df["Label"] = df["Label"].astype("int8")

# Keep numeric features only
features = df.drop(columns=["Label"])
features = features.select_dtypes(include=["number"])

# Replace any remaining invalid values
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features = features.fillna(0)

# Save cleaned dataset
cleaned = features.copy()
cleaned["Label"] = df["Label"]

output_path = OUT / "ddos_cleaned.csv"
cleaned.to_csv(output_path, index=False)

print("\n✅ Data preparation complete!")
print("Rows:", cleaned.shape[0])
print("Features:", features.shape[1])
print("Labels:")
print(cleaned["Label"].value_counts())
print("Saved to:", output_path)