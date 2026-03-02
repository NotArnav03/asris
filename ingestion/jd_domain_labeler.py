import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
META_DIR = BASE_DIR / "data" / "metadata"

# Load mapping
with open(META_DIR / "domain_mapping.json", "r") as f:
    domain_map = json.load(f)

# Load postings
postings = pd.read_csv(RAW_DIR / "postings.csv")

postings["title_lower"] = postings["title"].str.lower()

def assign_domain(title):
    for domain, keywords in domain_map.items():
        for kw in keywords:
            if kw in title:
                return domain
    return None

postings["assigned_domain"] = postings["title_lower"].apply(assign_domain)

# Keep only matched
filtered = postings[postings["assigned_domain"].notnull()]

print("Total JDs:", len(postings))
print("Matched JDs:", len(filtered))

print("\nDomain distribution:")
print(filtered["assigned_domain"].value_counts())

# Save filtered file
filtered.to_csv(RAW_DIR / "postings_labeled.csv", index=False)

print("\nLabeled dataset saved.")
