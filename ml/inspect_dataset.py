
from pathlib import Path
import pandas as pd

# Find CSV files inside data/raw
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw!")
    print("Current folder:", Path.cwd())
    print("Files in data/raw:", list(data_folder.iterdir()))
    raise SystemExit

# Use the first CSV found
data_path = csv_files[0]

print("Dataset found:", data_path)
print("Loading dataset...")

df = pd.read_csv(data_path, low_memory=False)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())

print("\nTraffic labels:")
if "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("Label column not found.")
    print("Available columns:", df.columns.tolist())