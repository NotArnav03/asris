import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"

df = pd.read_csv(RAW_DIR / "postings_labeled.csv")

balanced_list = []

for domain in df["assigned_domain"].unique():
    subset = df[df["assigned_domain"] == domain]
    sampled = subset.sample(n=min(110, len(subset)), random_state=42)
    balanced_list.append(sampled)

balanced_df = pd.concat(balanced_list)

print("Balanced JD distribution:")
print(balanced_df["assigned_domain"].value_counts())

balanced_df.to_csv(RAW_DIR / "postings_balanced.csv", index=False)

print("\nBalanced dataset saved.")
