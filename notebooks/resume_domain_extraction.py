from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_DIR = BASE_DIR / "data" / "raw" / "resumes"

domains = [folder.name for folder in RESUME_DIR.iterdir() if folder.is_dir()]

print("Resume Domains:")
for d in domains:
    print("-", d)
