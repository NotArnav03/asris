from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_DIR = BASE_DIR / "data" / "raw" / "resumes"

counts = {}

for folder in RESUME_DIR.iterdir():
    if folder.is_dir():
        file_count = len([f for f in folder.glob("*.pdf")])
        counts[folder.name] = file_count

print("Resume distribution:")
for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v}")
