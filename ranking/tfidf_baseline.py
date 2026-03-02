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

# Compute similarity
scores = []
labels = []

print("Computing similarity scores...")
for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    jd_vec = jd_vectors.get(row["job_id"])
    resume_file = row["resume_filename"].replace(".pdf", ".txt")
    resume_vec = resume_vectors.get(resume_file)

    if jd_vec is None or resume_vec is None:
        scores.append(0)
        labels.append(row["label"])
        continue

    sim = cosine_similarity(jd_vec, resume_vec)[0][0]
    scores.append(sim)
    labels.append(row["label"])

# Better threshold (use median)
threshold = sorted(scores)[int(len(scores) * 0.75)]
predictions = [1 if s > threshold else 0 for s in scores]

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))
