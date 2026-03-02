import os
import pandas as pd
import pdfplumber
from pathlib import Path
import re
from tqdm import tqdm

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESUME_DIR = BASE_DIR / "data" / "raw" / "resumes"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "resumes_cleaned"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# BASIC CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '', text)  # remove phone numbers
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text


# -----------------------------
# PROCESS CSV RESUMES
# -----------------------------
def process_csv():
    csv_path = RAW_RESUME_DIR / "Resume.csv"

    if not csv_path.exists():
        print("Resume.csv not found.")
        return

    df = pd.read_csv(csv_path)

    if "Resume_str" not in df.columns:
        print("Resume_str column not found.")
        return

    print("Processing CSV resumes...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        resume_text = str(row["Resume_str"])
        resume_text = clean_text(resume_text)

        output_file = PROCESSED_DIR / f"csv_resume_{idx}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(resume_text)


# -----------------------------
# PROCESS PDF RESUMES
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
    return text


def process_pdfs():
    print("Processing PDF resumes...")

    for root, dirs, files in os.walk(RAW_RESUME_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = Path(root) / file
                text = extract_text_from_pdf(pdf_path)
                text = clean_text(text)

                category = Path(root).name
                output_name = f"{category}_{file.replace('.pdf', '')}.txt"
                output_file = PROCESSED_DIR / output_name

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    process_csv()
    process_pdfs()
    print("Resume ingestion completed.")
