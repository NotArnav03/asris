# Push FAIMR to GitHub

## Step 1: Install Git
Download and install from: https://git-scm.com/download/win
(Keep all default options during install)

## Step 2: Create GitHub Repo
1. Go to https://github.com/new
2. Name it `faimr`
3. **Don't** check "Add a README" (you already have one)
4. Click "Create repository"

## Step 3: Push (run in Command Prompt, one by one)

```bash
cd C:\faimr
git init
```

### Add only the necessary files:
```bash
git add .gitignore
git add LICENSE
git add README.md
git add requirements.txt
git add config.py
git add run_pipeline.py

git add api/
git add embeddings/
git add evaluation/
git add experiments/
git add explainability/
git add fairness/
git add frontend/
git add ingestion/
git add notebooks/
git add preprocessing/
git add ranking/
git add tests/
```

### Commit and push:
```bash
git commit -m "FAIMR: AI Resume Screening with PDF upload & explainability"
git remote add origin https://github.com/YOUR_USERNAME/faimr.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

> If prompted for credentials, use a Personal Access Token:
> GitHub → Settings → Developer Settings → Personal Access Tokens → Generate New Token

## What's NOT pushed (by design)
- `data/` — 7,400+ dataset files (too large for Git)
- `__pycache__/` — Python bytecode
- Model weights, embedding caches, MLflow artifacts
- `PUSH_TO_GITHUB.md` — this file (not needed in repo)
