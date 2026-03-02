from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESUME_DIR = BASE_DIR / "data" / "raw" / "resumes"

records = []

for domain_folder in RAW_RESUME_DIR.iterdir():
    if domain_folder.is_dir():
        domain = domain_folder.name
        for pdf_file in domain_folder.glob("*.pdf"):
            records.append({
                "resume_filename": pdf_file.name,
                "domain": domain
            })

df = pd.DataFrame(records)

print("Total resumes:", len(df))
print(df["domain"].value_counts())

df.to_csv(BASE_DIR / "data" / "metadata" / "resume_metadata.csv", index=False)

print("\nResume metadata saved.")
