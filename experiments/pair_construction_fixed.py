"""
FAIMR — Fixed Pair Construction (Domain-Match Labeling)
Replaces the leaky skill-coverage-based labeling with domain-match labeling.
Resume domain is parsed from filename, JD domain from assigned_domain column.
Hard negatives are sampled from adjacent domains, not random.
Includes leakage verification via point-biserial correlation.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import pointbiserialr
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"
PROCESSED_RESUME_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"
LABELED_DIR = BASE_DIR / "data" / "labeled"
META_DIR = BASE_DIR / "data" / "metadata"

ADJACENT_DOMAINS = {
    "INFORMATION-TECHNOLOGY": ["ENGINEERING", "CONSULTANT", "DIGITAL-MEDIA", "BPO"],
    "ENGINEERING": ["INFORMATION-TECHNOLOGY", "CONSTRUCTION", "AUTOMOBILE", "AVIATION"],
    "FINANCE": ["BANKING", "ACCOUNTANT", "CONSULTANT", "BUSINESS-DEVELOPMENT"],
    "BANKING": ["FINANCE", "ACCOUNTANT", "SALES"],
    "ACCOUNTANT": ["FINANCE", "BANKING", "CONSULTANT"],
    "HR": ["BUSINESS-DEVELOPMENT", "CONSULTANT", "SALES", "PUBLIC-RELATIONS"],
    "SALES": ["BUSINESS-DEVELOPMENT", "BANKING", "HR", "PUBLIC-RELATIONS"],
    "HEALTHCARE": ["FITNESS", "AGRICULTURE", "TEACHER"],
    "DESIGNER": ["DIGITAL-MEDIA", "ARTS", "APPAREL"],
    "DIGITAL-MEDIA": ["DESIGNER", "ARTS", "PUBLIC-RELATIONS"],
    "ARTS": ["DESIGNER", "DIGITAL-MEDIA", "APPAREL"],
    "TEACHER": ["HR", "HEALTHCARE", "PUBLIC-RELATIONS"],
    "CONSULTANT": ["BUSINESS-DEVELOPMENT", "FINANCE", "INFORMATION-TECHNOLOGY"],
    "BUSINESS-DEVELOPMENT": ["SALES", "CONSULTANT", "HR", "FINANCE"],
    "PUBLIC-RELATIONS": ["DIGITAL-MEDIA", "HR", "SALES"],
    "ADVOCATE": ["CONSULTANT", "HR", "PUBLIC-RELATIONS"],
    "BPO": ["SALES", "HR", "INFORMATION-TECHNOLOGY"],
    "CHEF": ["FITNESS", "AGRICULTURE"],
    "CONSTRUCTION": ["ENGINEERING", "AUTOMOBILE"],
    "AUTOMOBILE": ["ENGINEERING", "CONSTRUCTION", "AVIATION"],
    "AVIATION": ["ENGINEERING", "AUTOMOBILE"],
    "AGRICULTURE": ["HEALTHCARE", "CHEF"],
    "FITNESS": ["HEALTHCARE", "CHEF"],
    "APPAREL": ["DESIGNER", "ARTS"],
}

POSITIVES_PER_JD = 5
NEGATIVES_PER_JD = 5


def extract_resume_domain(filename):
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0].upper()
    return "UNKNOWN"


def load_data():
    jds = pd.read_csv(RAW_DIR / "postings_balanced.csv")
    jds = jds.dropna(subset=["assigned_domain", "description"])
    jd_domains = dict(zip(jds["job_id"], jds["assigned_domain"].str.upper().str.strip()))
    jd_texts = dict(zip(jds["job_id"], jds["description"].astype(str)))

    resume_texts = {}
    resume_domains = {}
    for file in os.listdir(PROCESSED_RESUME_DIR):
        if file.endswith(".txt"):
            with open(PROCESSED_RESUME_DIR / file, "r", encoding="utf-8") as f:
                resume_texts[file] = f.read()
            resume_domains[file] = extract_resume_domain(file)

    resumes_by_domain = defaultdict(list)
    for fname, domain in resume_domains.items():
        resumes_by_domain[domain].append(fname)

    job_skills_path = RAW_DIR / "jobs" / "job_skills.csv"
    skills_map_path = RAW_DIR / "mappings" / "skills.csv"
    jd_skill_map = {}
    resume_skill_map = {}

    if job_skills_path.exists() and skills_map_path.exists():
        job_skills = pd.read_csv(job_skills_path)
        skills_map = pd.read_csv(skills_map_path)
        skill_dict = dict(zip(skills_map["skill_abr"], skills_map["skill_name"]))
        all_skill_names = [s.lower() for s in skills_map["skill_name"].tolist()]

        for _, row in job_skills.iterrows():
            job_id = row["job_id"]
            skill_abr = row["skill_abr"]
            if skill_abr in skill_dict:
                jd_skill_map.setdefault(job_id, set()).add(skill_dict[skill_abr].lower())

        for fname, text in resume_texts.items():
            text_lower = text.lower()
            matched = {s for s in all_skill_names if s in text_lower}
            resume_skill_map[fname] = matched

    return (jd_domains, jd_texts, resume_texts, resume_domains,
            resumes_by_domain, jd_skill_map, resume_skill_map)


def generate_pairs(jd_domains, jd_texts, resume_texts, resume_domains,
                   resumes_by_domain, jd_skill_map, resume_skill_map):
    pairs = []
    all_domains = set(jd_domains.values())

    sampled_jds = list(jd_domains.items())
    np.random.seed(42)
    np.random.shuffle(sampled_jds)

    for job_id, jd_domain in tqdm(sampled_jds, desc="Generating domain-match pairs"):
        if jd_domain == "UNKNOWN" or jd_domain not in all_domains:
            continue

        same_domain_resumes = resumes_by_domain.get(jd_domain, [])
        if len(same_domain_resumes) < POSITIVES_PER_JD:
            continue

        pos_sample = np.random.choice(same_domain_resumes, size=POSITIVES_PER_JD, replace=False)
        for resume_file in pos_sample:
            pairs.append({"job_id": job_id, "resume_filename": resume_file, "label": 1})

        adjacent = ADJACENT_DOMAINS.get(jd_domain, [])
        hard_neg_pool = []
        for adj_domain in adjacent:
            hard_neg_pool.extend(resumes_by_domain.get(adj_domain, []))

        if len(hard_neg_pool) < NEGATIVES_PER_JD:
            for d in all_domains:
                if d != jd_domain and d not in adjacent:
                    hard_neg_pool.extend(resumes_by_domain.get(d, []))

        if len(hard_neg_pool) >= NEGATIVES_PER_JD:
            neg_sample = np.random.choice(hard_neg_pool, size=NEGATIVES_PER_JD, replace=False)
            for resume_file in neg_sample:
                pairs.append({"job_id": job_id, "resume_filename": resume_file, "label": 0})

    return pd.DataFrame(pairs)


def verify_leakage(pairs_df, jd_skill_map, resume_skill_map):
    scr_scores = []
    labels = []

    for _, row in pairs_df.iterrows():
        job_id = row["job_id"]
        resume_file = row["resume_filename"]
        jd_skills = jd_skill_map.get(job_id, set())
        resume_skills = resume_skill_map.get(resume_file, set())

        if len(jd_skills) > 0:
            scr = len(jd_skills.intersection(resume_skills)) / len(jd_skills)
        else:
            scr = 0.0

        scr_scores.append(scr)
        labels.append(row["label"])

    corr, p_value = pointbiserialr(scr_scores, labels)
    return corr, p_value, scr_scores


if __name__ == "__main__":
    print("=" * 60)
    print("  FAIMR — Fixed Pair Construction (Domain-Match Labeling)")
    print("=" * 60)

    data = load_data()
    jd_domains, jd_texts, resume_texts, resume_domains, resumes_by_domain, jd_skill_map, resume_skill_map = data

    print(f"\nLoaded {len(jd_domains)} JDs, {len(resume_texts)} resumes")
    print(f"Resume domains: {len(set(resume_domains.values()))} unique")
    print(f"JD domains: {len(set(jd_domains.values()))} unique")

    pairs_df = generate_pairs(jd_domains, jd_texts, resume_texts, resume_domains,
                              resumes_by_domain, jd_skill_map, resume_skill_map)

    print(f"\nTotal pairs: {len(pairs_df)}")
    print(f"Label distribution:\n{pairs_df['label'].value_counts().to_string()}")

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELED_DIR / "domain_match_pairs.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    print("\n--- LEAKAGE VERIFICATION ---")
    corr, p_value, scr_scores = verify_leakage(pairs_df, jd_skill_map, resume_skill_map)
    print(f"Point-biserial correlation (SCR vs label): r = {corr:.4f}, p = {p_value:.6f}")
    print(f"Mean SCR (label=1): {np.mean([s for s, l in zip(scr_scores, pairs_df['label']) if l == 1]):.4f}")
    print(f"Mean SCR (label=0): {np.mean([s for s, l in zip(scr_scores, pairs_df['label']) if l == 0]):.4f}")

    if abs(corr) > 0.95:
        print("WARNING: SCR-label correlation > 0.95 — potential leakage still present!")
    elif abs(corr) > 0.4:
        print("OK: Moderate correlation — SCR is a legitimate predictive feature, not a label proxy.")
    else:
        print("OK: Low correlation — SCR provides weak but valid signal.")

    print("=" * 60)
