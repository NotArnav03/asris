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

# Load datasets
pairs = pd.read_csv(LABELED_DIR / "ranking_pairs.csv")
jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")

# Load SBERT model
print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build JD dictionary
jd_dict = dict(zip(jds["job_id"], jds["description"]))

# Load all resume texts
print("Loading resume texts...")
resume_texts = {}
for file in PROCESSED_RESUME_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        resume_texts[file.name] = f.read()

# Encode all JDs
print("Encoding JDs...")
jd_embeddings = {
    job_id: model.encode(str(text), show_progress_bar=False)
    for job_id, text in tqdm(jd_dict.items())
}

# Encode all resumes
print("Encoding resumes...")
resume_embeddings = {
    filename: model.encode(text, show_progress_bar=False)
    for filename, text in tqdm(resume_texts.items())
}

# Compute similarity.  Missing pairs are SKIPPED (not silently
# zeroed out as in a prior revision) — inserting 0 for a missing
# embedding biases ROC-AUC because the missing pair always appears
# at the bottom of the ranking and is treated as a "low score true
# negative" regardless of its real label.  Skipping is the correct
# behaviour: the pair contributes neither to scores nor labels.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from ranking.ranking_utils import classify_at_percentile  # noqa: E402

scores = []
labels = []
n_skipped = 0

print("Computing similarities...")
for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    jd_emb = jd_embeddings.get(row["job_id"])
    resume_file = row["resume_filename"].replace(".pdf", ".txt")
    resume_emb = resume_embeddings.get(resume_file)

    if jd_emb is None or resume_emb is None:
        n_skipped += 1
        continue

    sim = cosine_similarity([jd_emb], [resume_emb])[0][0]
    scores.append(sim)
    labels.append(row["label"])

if n_skipped:
    print(f"Skipped {n_skipped} pairs with missing embeddings.")

# 75th-percentile threshold — selects the top quartile.  Centralised
# through classify_at_percentile so every baseline uses the same
# threshold semantics (>=, not strict >).
threshold, predictions = classify_at_percentile(scores, percentile=75.0)
print(f"Threshold (75th pct): {threshold:.4f}")

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))
