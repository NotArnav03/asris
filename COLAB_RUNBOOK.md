# Colab Runbook — FAIMR Experiments

Run these cells in a **fresh Colab notebook** in order.
Each script produces numbers or LaTeX tables that replace values in the paper.

---

## Cell 1 — Clone the repo

```python
# Make sure we're at /content before cloning
import os
os.chdir("/content")
```

```bash
!git clone https://github.com/NotArnav03/asris.git
```

```python
# Move into the repo — all subsequent cells run from here
os.chdir("/content/asris")
print("CWD:", os.getcwd())   # should print /content/asris
```

---

## Cell 2 — Install dependencies

```bash
!pip install -q \
  sentence-transformers \
  xgboost \
  scikit-learn \
  scipy \
  pandas \
  numpy \
  tqdm \
  rank-bm25 \
  gensim \
  pdfplumber
```

---

## Cell 3 — Mount Drive and link data

Your `data/` folder must contain:
```
data/
  labeled/
    domain_match_pairs.csv
  processed/
    resumes_cleaned/        ← .txt resume files
  raw/
    job_descriptions/
      postings_balanced.csv
      jobs/job_skills.csv
      mappings/skills.csv
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import os, pathlib

# Set this to wherever your data folder lives on Drive
DRIVE_DATA = "/content/drive/MyDrive/asris_data/data"

repo_data = "/content/asris/data"

# If data is on Drive, symlink it (skip if you're uploading data directly)
if not os.path.exists(repo_data):
    os.symlink(DRIVE_DATA, repo_data)
    print("Symlinked:", repo_data, "->", DRIVE_DATA)
else:
    print("data/ already exists at", repo_data)

# Verify the critical files are reachable
checks = [
    "data/labeled/domain_match_pairs.csv",
    "data/processed/resumes_cleaned",
    "data/raw/job_descriptions/postings_balanced.csv",
]
for p in checks:
    exists = os.path.exists(p)
    print(f"  {'OK' if exists else 'MISSING'}  {p}")
```

> If any path shows **MISSING**, fix your Drive path in `DRIVE_DATA` before continuing.

---

## Cell 4 — Script 1: Counterfactual Re-evaluation  ⚠️ Run this first

This replaces the counterfactual numbers in the paper.
The old numbers were computed with a broken implementation.

**Expected runtime: 20–40 min on Colab CPU**

```bash
!python experiments/counterfactual_reeval.py 2>&1 | tee /tmp/cf_reeval.log
```

At the end you will see:
```
[ACTION REQUIRED] Update the following values in the paper:
  Abstract/Table: Mean |delta*| = X.XX
  Abstract/Table: Median latency = X.XX ms
  Abstract/Table: Greedy-optimal = XX.X%
```

Copy the printed LaTeX table into your paper to replace `\label{tab:counterfactual}`.

---

## Cell 5 — Script 2: Label Quality Validation

New table defending against *"domain-match is not a hiring-outcome label"*.
Add it to Section 3 (Data) or the Appendix.

**Expected runtime: 15–25 min**

```bash
!python experiments/label_quality_validation.py 2>&1 | tee /tmp/label_quality.log
```

You want to see:
```
[PASS] Domain-match labels are valid relevance proxies.
```

Copy the LaTeX table (`\label{tab:label_quality}`) into the paper.

---

## Cell 6 — Script 3: FCR Stress Test

The old version used random hash-based groups. This version uses the
gender detector on real resume text.

**Expected runtime: 5–10 min**

```bash
!python experiments/fcr_stress_test.py 2>&1 | tee /tmp/fcr_stress.log
```

Check the output for:
- **Detection coverage** — what % of resumes got a gender label
- Whether FCR restores AIR >= 0.8 in **all 5 folds** at 30% skew

Replace `\label{tab:fcr_stress}` and add `\label{tab:fcr_stress_folds}` in the paper.

> **If coverage is below ~30%:** your resume files have few gender signals
> (common for anonymised CVs). Add a sentence to the paper:
> *"Gender proxy detection achieved X% coverage; FCR correctness was
> additionally validated via synthetic group assignment (Section X)."*

---

## Cell 7 — Script 4: Ablation + Paired t-tests (if needed)

Only run this if you need to regenerate the ablation table.

```bash
!python experiments/ablation_stats.py 2>&1 | tee /tmp/ablation.log
```

---

## Cell 8 — Save outputs to Drive

```python
import shutil, os

output_dir = "/content/drive/MyDrive/asris_results"
os.makedirs(output_dir, exist_ok=True)

for log in [
    "/tmp/cf_reeval.log",
    "/tmp/label_quality.log",
    "/tmp/fcr_stress.log",
    "/tmp/ablation.log",
]:
    if os.path.exists(log):
        shutil.copy(log, output_dir)
        print("Saved:", log)
```

---

## Paper update checklist

After running all scripts, update the paper with fresh numbers:

- [ ] `\label{tab:counterfactual}` — replace with output from Script 1
- [ ] Abstract: *"averaging X.XX skills"* — update mean |δ*|
- [ ] Abstract: *"mean latency X.XX ms (median X.XX ms)"* — update latency
- [ ] Abstract: *"100% greedy-optimal match rate"* — update if changed
- [ ] `\label{tab:label_quality}` — insert new table from Script 2 (Section 3)
- [ ] `\label{tab:fcr_stress}` — replace with output from Script 3
- [ ] `\label{tab:fcr_stress_folds}` — insert per-fold table from Script 3
- [ ] FCR limitations: add detection coverage % from Script 3 output
