import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
LABELED_DIR = BASE_DIR / "data" / "labeled"

# Load data
job_skills = pd.read_csv(RAW_DIR / "jobs" / "job_skills.csv")
skills_map = pd.read_csv(RAW_DIR / "mappings" / "skills.csv")
jds = pd.read_csv(RAW_DIR / "postings_balanced.csv")

# Map skill_abr -> skill_name
skill_dict = dict(zip(skills_map["skill_abr"], skills_map["skill_name"]))

# Build JD -> skill_name list
jd_skill_map = {}

for _, row in job_skills.iterrows():
    job_id = row["job_id"]
    skill_abr = row["skill_abr"]

    if skill_abr in skill_dict:
        skill_name = skill_dict[skill_abr].lower()
        jd_skill_map.setdefault(job_id, set()).add(skill_name)

# Load resume texts
resume_texts = {}
for file in os.listdir(PROCESSED_RESUME_DIR):
    if file.endswith(".txt"):
        with open(PROCESSED_RESUME_DIR / file, "r", encoding="utf-8") as f:
            resume_texts[file] = f.read().lower()

# Extract resume skills
resume_skill_map = {}

all_skill_names = [s.lower() for s in skills_map["skill_name"].tolist()]

for filename, text in resume_texts.items():
    matched_skills = set()
    for skill in all_skill_names:
        if skill in text:
            matched_skills.add(skill)
    resume_skill_map[filename] = matched_skills

pairs = []

print("Generating skill-based pairs...")

for job_id, jd_skills in tqdm(jd_skill_map.items()):
    if job_id not in jds["job_id"].values:
        continue

    scores = []

    for resume_file, resume_skills in resume_skill_map.items():
        if len(jd_skills) == 0:
            continue

        overlap = len(jd_skills.intersection(resume_skills))
        coverage = overlap / len(jd_skills)

        scores.append((resume_file, coverage))

    if not scores:
        continue

    scores.sort(key=lambda x: x[1])

    top_k = 5
    bottom_k = 5

    positives = scores[-top_k:]
    negatives = scores[:bottom_k]

    for resume_file, _ in positives:
        pairs.append({
            "job_id": job_id,
            "resume_filename": resume_file,
            "label": 1
        })

    for resume_file, _ in negatives:
        pairs.append({
            "job_id": job_id,
            "resume_filename": resume_file,
            "label": 0
        })

pairs_df = pd.DataFrame(pairs)

print("Total skill-based pairs:", len(pairs_df))
print(pairs_df["label"].value_counts())

pairs_df.to_csv(LABELED_DIR / "skill_based_pairs.csv", index=False)

print("Skill-based ranking dataset saved.")