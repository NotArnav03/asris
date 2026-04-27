# Colab Runbook — FAIMR Experiments

Run these cells **in order** in a fresh Colab notebook.

---

## Cell 1 — Clone and enter repo

```python
import os
os.chdir("/content")
```

```bash
!git clone https://github.com/NotArnav03/asris.git
```

```python
os.chdir("/content/asris")
print("CWD:", os.getcwd())   # must print /content/asris
```

```bash
!git pull origin main        # get the latest fixes
```

---

## Cell 2 — Install dependencies

```bash
!pip install -q \
  sentence-transformers xgboost scikit-learn scipy \
  pandas numpy tqdm rank-bm25 gensim pdfplumber
```

---

## Cell 3 — Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## Cell 4 — Link your data  ⚠️ Read this carefully

The scripts expect this layout under `/content/asris/data/`:

```
data/
  labeled/
    domain_match_pairs.csv
  processed/
    resumes_cleaned/        ← one .txt file per resume
  raw/
    job_descriptions/
      postings_balanced.csv
      jobs/
        job_skills.csv
      mappings/
        skills.csv
```

**Option A — your data is already on Google Drive**

```python
import os

# Change this to wherever your data folder is on Drive
DRIVE_DATA = "/content/drive/MyDrive/asris_data/data"

repo_data = "/content/asris/data"

if os.path.islink(repo_data):
    os.unlink(repo_data)          # remove stale symlink if any

if not os.path.exists(repo_data):
    os.symlink(DRIVE_DATA, repo_data)
    print("Linked:", repo_data, "->", DRIVE_DATA)
else:
    print("data/ already exists — skipping symlink")
```

**Option B — upload a zip from your local machine**

```python
from google.colab import files
uploaded = files.upload()   # pick your data.zip in the dialog
```

```bash
!unzip -q data.zip -d /content/asris/
```

---

## Cell 5 — Verify data is reachable  ← run this before anything else

```python
import os

required = {
    "Labeled pairs":      "data/labeled/domain_match_pairs.csv",
    "Resumes dir":        "data/processed/resumes_cleaned",
    "JD postings":        "data/raw/job_descriptions/postings_balanced.csv",
    "Job skills":         "data/raw/job_descriptions/jobs/job_skills.csv",
    "Skills map":         "data/raw/job_descriptions/mappings/skills.csv",
}

all_ok = True
for name, path in required.items():
    ok = os.path.exists(path)
    print(f"  {'OK  ' if ok else 'MISS'} {name:20s}  {path}")
    if not ok:
        all_ok = False

if all_ok:
    print("\n  All data files found. Ready to run experiments.")
else:
    print("\n  Fix the MISSING paths above before continuing.")
```

**Only proceed once all lines print `OK`.**

---

## Cell 6 — Script 1: Counterfactual Re-evaluation  ⚠️ Run first

Replaces the counterfactual numbers currently in the paper
(mean |δ*|, greedy-optimal rate, latency). The old numbers were from
a broken implementation.

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

Copy the LaTeX table into your paper to replace `\label{tab:counterfactual}`.

---

## Cell 7 — Script 2: Label Quality Validation

Produces a new table defending the domain-match labels against the
reviewer concern *"this is not a hiring-outcome label"*.
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

## Cell 8 — Script 3: FCR Stress Test

The old version used random hash-based groups. This uses the gender
detector on real resume text.

**Expected runtime: 5–10 min**

```bash
!python experiments/fcr_stress_test.py 2>&1 | tee /tmp/fcr_stress.log
```

Replace `\label{tab:fcr_stress}` and add `\label{tab:fcr_stress_folds}` in the paper.

> **If coverage is below ~30%:** add a sentence to the paper:
> *"Gender proxy detection achieved X% coverage on this corpus; FCR
> correctness was additionally validated via synthetic group assignment."*

---

## Cell 9 — Save logs to Drive

```python
import shutil, os

out = "/content/drive/MyDrive/asris_results"
os.makedirs(out, exist_ok=True)

for log in ["/tmp/cf_reeval.log", "/tmp/label_quality.log", "/tmp/fcr_stress.log"]:
    if os.path.exists(log):
        shutil.copy(log, out)
        print("Saved:", log)
```

---

## Paper update checklist

- [ ] `\label{tab:counterfactual}` — replace with Script 1 output
- [ ] Abstract: *"averaging X.XX skills"* — update mean |δ*|
- [ ] Abstract: *"mean latency X.XX ms (median X.XX ms)"* — update latency
- [ ] Abstract: *"100% greedy-optimal match rate"* — update if changed
- [ ] `\label{tab:label_quality}` — insert new table from Script 2 (Section 3)
- [ ] `\label{tab:fcr_stress}` — replace with Script 3 output
- [ ] `\label{tab:fcr_stress_folds}` — insert per-fold table from Script 3
- [ ] FCR limitations: add detection coverage % from Script 3 output
