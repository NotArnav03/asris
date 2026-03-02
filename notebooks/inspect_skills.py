import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "job_descriptions"

job_skills = pd.read_csv(RAW_DIR / "jobs" / "job_skills.csv")
skills_map = pd.read_csv(RAW_DIR / "mappings" / "skills.csv")

print("job_skills columns:")
print(job_skills.columns.tolist())

print("\nskills_map columns:")
print(skills_map.columns.tolist())

print("\nSample job_skills:")
print(job_skills.head())

print("\nSample skills_map:")
print(skills_map.head())