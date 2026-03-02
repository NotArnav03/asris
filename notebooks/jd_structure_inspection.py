import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"

postings = pd.read_csv(RAW_DIR / "postings.csv")

print("Columns:")
print(postings.columns.tolist())

print("\nSample rows:")
print(postings.head())
