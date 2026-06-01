"""FAIMR Plus -- RoBERTa + INLP plugin for Bias in Bios.

This plugin reproduces the INLP debiasing protocol (Ravfogel 2020) on
top of a RoBERTa-base occupation classifier fine-tuned on the Bias in
Bios training set.  The output: a per-bio occupation prediction whose
gender-leakage subspace has been iteratively null-space-projected
away.

The notebook `train_colab.ipynb` is the user-facing entry point.  It
runs end-to-end on a free Colab T4 GPU in ~30-45 minutes:

  1. Download Bias in Bios train + test parquet from HuggingFace
  2. Fine-tune RoBERTa-base on the 28-occupation task (1 epoch)
  3. Extract [CLS] embeddings for every bio
  4. Iteratively fit a gender LR on the embeddings, take its null
     space, project embeddings onto it, repeat until gender accuracy
     -> 0.5 (chance)
  5. Re-train the occupation head on the debiased embeddings
  6. Evaluate per-occupation TPR gap by gender on the test set
  7. Upload trained weights + results.json back to the FAIMR repo

The output artefact (`weights.pt`, occupation head + projection
matrix) is small enough to commit to the repo (~1 MB).

The runtime inference path (predict.py) takes a list of bios, runs
them through the frozen RoBERTa, applies the debiasing projection,
and returns occupation predictions + TPR gap.  This plugin requires
torch and transformers.
"""
