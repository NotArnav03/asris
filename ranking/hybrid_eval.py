import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
LABELED_DIR = BASE_DIR / "data" / "labeled"

# Load skill-based pairs
pairs = pd.read_csv(LABELED_DIR / "skill_based_pairs.csv")

# Load JD data
jds = pd.read_csv(RAW_DIR / "postings_balanced.csv")

# Load skill mappings
job_skills = pd.read_csv(RAW_DIR / "jobs" / "job_skills.csv")
skills_map = pd.read_csv(RAW_DIR / "mappings" / "skills.csv")

# Build skill lookup
skill_dict = dict(zip(skills_map["skill_abr"], skills_map["skill_name"]))

jd_skill_map = {}
for _, row in job_skills.iterrows():
    job_id = row["job_id"]
    skill_abr = row["skill_abr"]
    if skill_abr in skill_dict:
        skill_name = skill_dict[skill_abr].lower()
        jd_skill_map.setdefault(job_id, set()).add(skill_name)

# Load resume texts
resume_texts = {}
for file in PROCESSED_RESUME_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        resume_texts[file.name] = f.read().lower()

# Extract resume skills
all_skill_names = [s.lower() for s in skills_map["skill_name"].tolist()]
resume_skill_map = {}

for filename, text in resume_texts.items():
    matched = set()
    for skill in all_skill_names:
        if skill in text:
            matched.add(skill)
    resume_skill_map[filename] = matched

# Load SBERT
print("Loading SBERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode JDs
jd_dict = dict(zip(jds["job_id"], jds["description"]))
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

print("Computing hybrid scores...")

alpha = 0.7
beta = 0.3

for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
    job_id = row["job_id"]
    resume_file = row["resume_filename"]

    jd_emb = jd_embeddings.get(job_id)
    resume_emb = resume_embeddings.get(resume_file)

    if jd_emb is None or resume_emb is None:
        continue

    # SBERT similarity
    sbert_score = cosine_similarity([jd_emb], [resume_emb])[0][0]

    # Normalize SBERT score to [0,1]
    sbert_score = (sbert_score + 1) / 2

    # Skill coverage
    jd_skills = jd_skill_map.get(job_id, set())
    resume_skills = resume_skill_map.get(resume_file, set())

    if len(jd_skills) > 0:
        coverage = len(jd_skills.intersection(resume_skills)) / len(jd_skills)
    else:
        coverage = 0

    final_score = alpha * sbert_score + beta * coverage

    scores.append(final_score)
    labels.append(row["label"])

threshold = np.median(scores)
predictions = [1 if s > threshold else 0 for s in scores]

print("\nClassification Report:")
print(classification_report(labels, predictions))

print("ROC-AUC Score:", roc_auc_score(labels, scores))