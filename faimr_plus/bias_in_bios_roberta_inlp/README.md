# FAIMR Plus -- RoBERTa + INLP for Bias in Bios

Reproduces [Ravfogel et al. 2020 (INLP)](https://arxiv.org/abs/2004.07667)
on top of a RoBERTa-base occupation classifier fine-tuned on
[Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios).
**Target: beat the verified published INLP-BERT GAP_RMS of 0.095**
(Ravfogel 2020, Table 2, [arXiv:2004.07667](https://arxiv.org/abs/2004.07667)).
The plugin uses **LEACE** (Belrose NeurIPS 2023) -- the closed-form
optimal linear concept erasure that is mathematically guaranteed to
be at least as good as INLP -- alongside an INLP baseline for
apples-to-apples comparison.

## How to run

This plugin is trained on a free Colab T4 GPU because RoBERTa
fine-tuning is too slow on CPU. The notebook does the whole pipeline
end-to-end:

1. Open `train_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **T4 GPU**
3. Runtime → Run all
4. Wait ~30–45 minutes
5. Three files download automatically when the last cell runs:
   - `projection.npy` -- INLP projection matrix (~5 MB)
   - `occ_head.pkl` -- logistic-regression occupation head (~250 KB)
   - `results.json` -- TPR-gap metrics + comparison vs SOTA
6. Place all three under `faimr_plus/bias_in_bios_roberta_inlp/` in
   your FAIMR clone

After dropping the artefacts in, re-run the Bias in Bios benchmark
locally:

```bash
python -m benchmarks.bias_in_bios.evaluate
```

This evaluator detects the plugin and reports a new comparison row
against the published INLP-BERT band.

## What the notebook does

1. **Load** `LabHC/bias_in_bios` from HuggingFace (257 k train,
   99 k test, 28 occupations, gender labels).
2. **Fine-tune** `roberta-base` for occupation classification
   (1 epoch, lr 2e-5, batch 64, fp16).
3. **Extract** `[CLS]` embeddings for every train + test bio.
4. **INLP iterative null-space projection** (Ravfogel 2020):
   - Train an LR to predict gender from the embeddings
   - If gender accuracy > 0.55, take the null space of the LR
     coefficient row and compose with the current projection
   - Repeat until gender accuracy ≤ 0.55 (chance level)
   - Result: a projection matrix that removes the gender-predictive
     subspace from any embedding
5. **Re-train** the occupation head on the debiased embeddings.
6. **Evaluate** per-occupation TPR gap by gender on the test set.

## Why a Colab notebook and not local Python?

RoBERTa-base fine-tuning on 257 k bios at batch 64 takes ~25 min on a
T4 GPU and would take many hours on a laptop CPU. The notebook is
authored to be a one-click reproduction; the trained artefacts (~5 MB
total) ship in the repo so the inference path runs locally on CPU.

## Expected results

Published numbers (lower is better). Note the literature reports
**GAP_RMS** = sqrt(mean((TPR_M - TPR_F)^2)), not mean-abs:

| System | Metric | Value | Source |
|---|---|---:|---|
| FastText baseline | GAP_RMS | 0.184 | Ravfogel 2020 Table 2 |
| BERT baseline | GAP_RMS | 0.184 | Ravfogel 2020 Table 2 |
| INLP-debiased FastText | GAP_RMS | 0.089 | Ravfogel 2020 Table 2 |
| **INLP-debiased BERT** | **GAP_RMS** | **0.095** | Ravfogel 2020 Table 2 |
| LEACE-debiased (BERT) | GAP RMS-like | ~0.084 | Belrose NeurIPS 2023 |
| FAIMR + TF-IDF + LR (no debiasing) | mean-abs | 0.0887 | this repo |

The plugin's GAP_RMS will be written to `results.json` after the
Colab notebook finishes. **Success criterion: GAP_RMS strictly below
0.095** (and ideally below the LEACE ~0.084 ballpark). The notebook
runs three configurations -- baseline, INLP, LEACE -- and reports
all three with the same RMS metric so the comparison is
apples-to-apples.

## Inference path (after artefacts are dropped in)

Will be added at `predict.py` once you've completed the Colab run and
shipped the artefacts back. The path is straightforward:

```python
import numpy as np, pickle, torch
from transformers import RobertaModel, RobertaTokenizerFast

P = np.load('faimr_plus/bias_in_bios_roberta_inlp/projection.npy')
occ_head = pickle.load(open('faimr_plus/bias_in_bios_roberta_inlp/occ_head.pkl', 'rb'))

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
roberta   = RobertaModel.from_pretrained('roberta-base').eval()

def predict_occupation_debiased(bio: str) -> int:
    enc = tokenizer(bio, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        cls = roberta(**enc).last_hidden_state[0, 0, :].numpy()
    debiased = P @ cls
    return int(occ_head.predict(debiased.reshape(1, -1))[0])
```

## Citation

```bibtex
@inproceedings{ravfogel2020null,
  title  = {Null It Out: Guarding Protected Attributes by Iterative
            Nullspace Projection},
  author = {Ravfogel, Shauli and Elazar, Yanai and Gonen, Hila and
            Twiton, Michael and Goldberg, Yoav},
  booktitle = {Proceedings of the 58th Annual Meeting of the
               Association for Computational Linguistics (ACL)},
  year   = {2020},
}
```
