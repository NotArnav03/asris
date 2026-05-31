import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
RAW_JD_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
LABELED_DIR = BASE_DIR / "data" / "labeled"

# Load datasets
pairs = pd.read_csv(LABELED_DIR / "ranking_pairs.csv")
jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")

# JD dictionary
jd_dict = dict(zip(jds["job_id"], jds["description"]))

# Load all resume texts
print("Loading resume texts...")
resume_texts = {}
for file in os.listdir(PROCESSED_RESUME_DIR):
    if file.endswith(".txt"):
        with open(PROCESSED_RESUME_DIR / file, "r", encoding="utf-8") as f:
            resume_texts[file] = f.read()

# Build full corpus
print("Building corpus...")
corpus = []

# Add all JD texts
for jd in jd_dict.values():
    corpus.append(str(jd))

# Add all resume texts
for text in resume_texts.values():
    corpus.append(str(text))

# Fit global TF-IDF
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(corpus)

# Precompute JD vectors
print("Transforming JD texts...")
jd_vectors = {
    job_id: vectorizer.transform([str(text)])
    for job_id, text in jd_dict.items()
}

# Precompute Resume vectors
print("Transforming Resume texts...")
resume_vectors = {
    filename: vectorizer.transform([text])
    for filename, text in resume_texts.items()
}

# Compute similarity.  Missing pairs are SKIPPED — see sbert_baseline.py
# for the rationale.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from ranking.ranking_utils import classify_at_percentile  # noqa: E402

scores = []
labels = []
n_skipped = 0

print("Computing similarity scores...")
for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    jd_vec = jd_vectors.get(row["job_id"])
    resume_file = row["resume_filename"].replace(".pdf", ".txt")
    resume_vec = resume_vectors.get(resume_file)

    if jd_vec is None or resume_vec is None:
        n_skipped += 1
        continue

    sim = cosine_similarity(jd_vec, resume_vec)[0][0]
    scores.append(sim)
    labels.append(row["label"])

if n_skipped:
    print(f"Skipped {n_skipped} pairs with missing vectors.")

# 75th-percentile threshold — selects the top quartile.  Prior code
# had a misleading "use median" comment but used 75% — fixed both
# the threshold semantics (>= via classify_at_percentile) and the
# comment so they agree.
threshold, predictions = classify_at_percentile(scores, percentile=75.0)
print(f"Threshold (75th pct): {threshold:.4f}")

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))
