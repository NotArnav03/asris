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

# Compute similarity
scores = []
labels = []

print("Computing similarities...")
for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    jd_emb = jd_embeddings.get(row["job_id"])
    resume_file = row["resume_filename"].replace(".pdf", ".txt")
    resume_emb = resume_embeddings.get(resume_file)

    if jd_emb is None or resume_emb is None:
        scores.append(0)
        labels.append(row["label"])
        continue

    sim = cosine_similarity([jd_emb], [resume_emb])[0][0]
    scores.append(sim)
    labels.append(row["label"])

# Use percentile threshold
threshold = sorted(scores)[int(len(scores) * 0.75)]
predictions = [1 if s > threshold else 0 for s in scores]

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))
