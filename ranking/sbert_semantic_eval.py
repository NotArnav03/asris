import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
RAW_JD_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
LABELED_DIR = BASE_DIR / "data" / "labeled"

# Load data
pairs = pd.read_csv(LABELED_DIR / "semantic_ranking_pairs.csv")
jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

jd_dict = dict(zip(jds["job_id"], jds["description"]))

# Load resume texts
resume_texts = {}
for file in PROCESSED_RESUME_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        resume_texts[file.name] = f.read()

# Encode JDs
print("Encoding JDs...")
jd_embeddings = {
    job_id: model.encode(str(text), show_progress_bar=False)
    for job_id, text in tqdm(jd_dict.items())
}

# Encode resumes
print("Encoding resumes...")
resume_embeddings = {
    filename: model.encode(text, show_progress_bar=False)
    for filename, text in tqdm(resume_texts.items())
}

scores = []
labels = []

print("Computing similarities...")
for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    jd_emb = jd_embeddings.get(row["job_id"])
    resume_file = row["resume_filename"]
    resume_emb = resume_embeddings.get(resume_file)

    if jd_emb is None or resume_emb is None:
        continue

    sim = cosine_similarity([jd_emb], [resume_emb])[0][0]
    scores.append(sim)
    labels.append(row["label"])

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from ranking.ranking_utils import classify_at_percentile  # noqa: E402

# Median threshold (50th percentile) with >= boundary semantics.
threshold, predictions = classify_at_percentile(scores, percentile=50.0)
print(f"Threshold (median): {threshold:.4f}")

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))