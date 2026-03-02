import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_JD_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
META_DIR = BASE_DIR / "data" / "metadata"

# Load data
jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")
resumes = pd.read_csv(META_DIR / "resume_metadata.csv")

pairs = []

for _, jd in jds.iterrows():
    jd_domain = jd["assigned_domain"]

    # Positive resume (same domain)
    positive_resumes = resumes[resumes["domain"] == jd_domain]

    if len(positive_resumes) == 0:
        continue

    pos_resume = positive_resumes.sample(1).iloc[0]

    pairs.append({
        "job_id": jd["job_id"],
        "resume_filename": pos_resume["resume_filename"],
        "jd_domain": jd_domain,
        "resume_domain": pos_resume["domain"],
        "label": 1
    })

    # Negative resumes (3 from other domains)
    negative_resumes = resumes[resumes["domain"] != jd_domain]

    neg_samples = negative_resumes.sample(3)

    for _, neg_resume in neg_samples.iterrows():
        pairs.append({
            "job_id": jd["job_id"],
            "resume_filename": neg_resume["resume_filename"],
            "jd_domain": jd_domain,
            "resume_domain": neg_resume["domain"],
            "label": 0
        })

pairs_df = pd.DataFrame(pairs)

print("Total training pairs:", len(pairs_df))
print("\nLabel distribution:")
print(pairs_df["label"].value_counts())

pairs_df.to_csv(BASE_DIR / "data" / "labeled" / "ranking_pairs.csv", index=False)

print("\nPair dataset saved.")
