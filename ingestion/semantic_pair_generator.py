import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
RAW_JD_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
LABELED_DIR = BASE_DIR / "data" / "labeled"

# Load balanced JDs
jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")

# Load resume texts
resume_texts = {}
for file in os.listdir(PROCESSED_RESUME_DIR):
    if file.endswith(".txt"):
        with open(PROCESSED_RESUME_DIR / file, "r", encoding="utf-8") as f:
            resume_texts[file] = f.read()

resume_files = list(resume_texts.keys())
resume_corpus = list(resume_texts.values())

# Fit global TF-IDF on all resumes + JDs
print("Fitting TF-IDF...")
corpus = resume_corpus + list(jds["description"].astype(str))
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(corpus)

resume_matrix = vectorizer.transform(resume_corpus)

pairs = []

print("Generating semantic pairs...")

for _, jd in tqdm(jds.iterrows(), total=len(jds)):
    jd_text = str(jd["description"])
    jd_vec = vectorizer.transform([jd_text])

    sims = cosine_similarity(jd_vec, resume_matrix)[0]

    sorted_indices = np.argsort(sims)

    top_k = 10
    bottom_k = 10

    positive_indices = sorted_indices[-top_k:]
    negative_indices = sorted_indices[:bottom_k]

    for idx in positive_indices:
        pairs.append({
            "job_id": jd["job_id"],
            "resume_filename": resume_files[idx],
            "label": 1
        })

    for idx in negative_indices:
        pairs.append({
            "job_id": jd["job_id"],
            "resume_filename": resume_files[idx],
            "label": 0
        })

pairs_df = pd.DataFrame(pairs)

print("Total semantic pairs:", len(pairs_df))
print(pairs_df["label"].value_counts())

pairs_df.to_csv(LABELED_DIR / "semantic_ranking_pairs.csv", index=False)

print("Semantic ranking dataset saved.")
