# Colab Runbook — FAIMR Experiments

Run these in order before submitting. Each script produces numbers or
LaTeX tables that need to replace values currently in the paper.

---

## 0. Setup (run once at the top of your notebook)

```python
# Mount Drive if your data is there
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Clone the repo (or upload it)
!git clone https://github.com/<your-username>/asris.git
%cd asris
```

```bash
# Install dependencies
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
  spacy \
  pdfplumber \
  fastapi \
  uvicorn
```

```bash
# If your data lives on Drive, symlink it so paths resolve
!ln -s /content/drive/MyDrive/asris_data/data data
```

Your `data/` folder must contain:
```
data/
  labeled/
    domain_match_pairs.csv
  processed/
    resumes_cleaned/          ← .txt files
  raw/
    job_descriptions/
      postings_balanced.csv
      jobs/job_skills.csv
      mappings/skills.csv
```

---

## 1. Counterfactual Re-evaluation  ⚠️ MUST RUN BEFORE SUBMITTING

This replaces the counterfactual numbers in the paper (mean |δ*|,
greedy-optimal rate, latency). The old numbers were computed with a
broken implementation. These are the correct ones.

**Expected runtime: 20–40 min on Colab CPU (26k pairs + 8k candidates)**

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

Copy the LaTeX table it prints into your paper to replace `\label{tab:counterfactual}`.

---

## 2. Label Quality Validation

This is a new table that defends against the reviewer concern
*"domain-match is not a hiring-outcome label"*. Put it in Section 3
(Data) or the Appendix.

**Expected runtime: 15–25 min (SBERT encoding of all pairs)**

```bash
!python experiments/label_quality_validation.py 2>&1 | tee /tmp/label_quality.log
```

You want to see:
```
All signals positively correlated with label: YES
All correlations statistically significant (p<0.05): YES
[PASS] Domain-match labels are valid relevance proxies.
```

If any signal shows `[WARN]`, investigate before claiming label validity.

Copy the printed LaTeX table (`\label{tab:label_quality}`) into the paper.

---

## 3. FCR Stress Test (updated with real gender groups)

The old version used `hash(filename) % 2` for group assignment, which
is random noise. The new version uses the gender detector on actual
resume text, so the results are meaningful.

**Expected runtime: 5–10 min**

```bash
!python experiments/fcr_stress_test.py 2>&1 | tee /tmp/fcr_stress.log
```

Check the output for:
- Detection coverage (what % of resumes got a gender label)
- Whether FCR restores AIR >= 0.8 in **all 5 folds** at 30% skew

Replace `\label{tab:fcr_stress}` in the paper with the new LaTeX tables.

> **Note:** If coverage is below ~30%, the gender detector found few
> gender signals in your resume files (common for anonymised CVs).
> In that case, add a sentence to the paper: *"Gender proxy detection
> achieved X% coverage on this corpus; FCR correctness was validated
> via synthetic group assignment on the remaining candidates."*

---

## 4. Full Ablation + Paired t-tests (if not already cached)

If you need to regenerate the ablation table and significance tests:

```bash
!python experiments/ablation_stats.py 2>&1 | tee /tmp/ablation.log
```

This also re-runs the model-based counterfactual loop internally
(redundant with Script 1, but included for completeness).

---

## 5. Baseline Comparison (if not already cached)

Only needed if your `data/baseline_cache/` folder is empty or missing.
This is the slowest script (~45–90 min, dominated by Cross-Encoder).

```bash
!python experiments/baseline_comparison.py 2>&1 | tee /tmp/baselines.log
```

---

## Saving outputs back to Drive

```python
import shutil, os

output_dir = "/content/drive/MyDrive/asris_results"
os.makedirs(output_dir, exist_ok=True)

for log in ["/tmp/cf_reeval.log", "/tmp/label_quality.log",
            "/tmp/fcr_stress.log", "/tmp/ablation.log"]:
    if os.path.exists(log):
        shutil.copy(log, output_dir)

print("Logs saved to Drive.")
```

---

## Paper update checklist

After running all scripts, update the paper with fresh numbers:

- [ ] `\label{tab:counterfactual}` — replace with output from Script 1
- [ ] Abstract: *"averaging X.XX skills"* — update mean |δ*|
- [ ] Abstract: *"mean latency X.XX ms (median X.XX ms)"* — update latency
- [ ] Abstract: *"100% greedy-optimal match rate"* — update if it changed
- [ ] `\label{tab:label_quality}` — insert new table from Script 2 (Section 3)
- [ ] `\label{tab:fcr_stress}` — replace with output from Script 3
- [ ] `\label{tab:fcr_stress_folds}` — insert new per-fold table from Script 3
- [ ] FCR limitations section: add detection coverage % from Script 3 output
