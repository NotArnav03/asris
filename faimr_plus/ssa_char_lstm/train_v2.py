"""SSA char-LSTM v2 -- larger training set + larger architecture to
strictly beat Hu 2021's published char-LSTM SOTA (0.940 accuracy on
the Yahoo->SSA protocol).

Protocol:
  - Train set: FAIMR's `data/names/training_corpus.csv`
    (~45k names, firstname-database-derived, multicultural).  This
    is comparable in scale and provenance to Hu's Yahoo training
    set -- both are "everywhere on the web except SSA".
  - Validation set: 10% hold-out from the training corpus (hashed
    by name for determinism).
  - Test set: the FULL SSA aggregate from
    `benchmarks/ssa_name_gender/load.py`.  This matches Hu 2021's
    Yahoo->SSA evaluation protocol exactly: a completely-unseen
    corpus.

Architecture (bigger than v1, since train set is ~8x larger):
  - char embedding 64
  - 2-layer bi-LSTM hidden 128
  - max-pool + last-state -> 512-d
  - MLP head: 512 -> 128 -> 1, GELU + dropout 0.4

Training:
  - Hard labels with 0.05 smoothing
  - Sample weights = sqrt(corpus weight) so well-attested names
    contribute more
  - AdamW + cosine LR schedule, 60 epochs, patience 10

Determinism: seed 20251128.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from faimr_plus.ssa_char_lstm.model import (
    CharBiLSTMGenderClassifier, MAX_LEN, encode_name,
)

SEED = 20251128
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights_v2.pt"
META_PATH = ROOT / "meta_v2.json"
TRAINING_CORPUS = REPO_ROOT / "data" / "names" / "training_corpus.csv"

BATCH_SIZE = 256
N_EPOCHS = 60
LR = 1.5e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 5e-4
PATIENCE = 10
LABEL_SMOOTH = 0.05

# v2 architecture
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4


def _deterministic_split(name: str, frac_train: float = 0.9) -> str:
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:4]
    n = int(h, 16) / 0xffff
    return "train" if n < frac_train else "val"


def _load_training_corpus():
    import pandas as pd
    df = pd.read_csv(
        TRAINING_CORPUS, keep_default_na=False, na_values=[""],
    )
    df["name"] = df["name"].astype(str).str.lower()
    # Use only names with reasonably clear gender labels.
    df = df[(df["p_female"] >= 0.7) | (df["p_female"] <= 0.3)].copy()
    df["y"] = (df["p_female"] >= 0.5).astype(np.float32)
    df["weight"] = df["weight"].astype(float).clip(lower=0.5)
    df["split"] = df["name"].apply(_deterministic_split)
    return df


def _build_tensors(df, split: str):
    sub = df[df["split"] == split]
    X = np.array(
        [encode_name(n) for n in sub["name"].tolist()],
        dtype=np.int64,
    )
    hard = sub["y"].to_numpy(dtype=np.float32)
    y = hard * (1 - LABEL_SMOOTH) + (1 - hard) * LABEL_SMOOTH
    w = np.sqrt(sub["weight"].to_numpy(dtype=np.float32))
    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


def main() -> int:
    print(f"# SSA char-LSTM v2 training (seed={SEED})")
    print()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading training corpus from {TRAINING_CORPUS.relative_to(REPO_ROOT)} ...")
    df = _load_training_corpus()
    print(f"  total clean-label names: {len(df)}")
    for s in ("train", "val"):
        print(f"  {s:<5}  n={int((df['split'] == s).sum())}")
    print()

    Xtr, ytr, wtr = _build_tensors(df, "train")
    Xva, yva, wva = _build_tensors(df, "val")

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr, wtr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = CharBiLSTMGenderClassifier(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_acc = 0.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_n = 0
        for xb, yb, wb in train_loader:
            logits = model(xb)
            losses = loss_fn(logits, yb)
            loss = (losses * wb).sum() / wb.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * xb.size(0)
            total_n += xb.size(0)
        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs >= 0.5).long()
            val_labels = (yva >= 0.5).long()
            val_acc = float((val_preds == val_labels).float().mean())
            val_loss = float(loss_fn(val_logits, yva).mean())

        elapsed = time.time() - t0
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            epochs_since_improvement = 0
            marker = " *"
        else:
            epochs_since_improvement += 1

        cur_lr = scheduler.get_last_lr()[0]
        print(f"  epoch {epoch:>2}  lr={cur_lr:.1e}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({elapsed:.1f}s){marker}")

        scheduler.step()
        if epochs_since_improvement >= PATIENCE:
            print(f"  early stop (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print()
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Wrote {WEIGHTS_PATH.relative_to(REPO_ROOT)}  "
          f"({WEIGHTS_PATH.stat().st_size} bytes)")

    # -------------------------------------------------------------
    # Test on SSA corpus (Hu 2021 Yahoo->SSA protocol).
    # -------------------------------------------------------------
    print()
    print("## Test on full SSA corpus (Hu 2021 Yahoo->SSA protocol)")
    from benchmarks.ssa_name_gender.load import load_per_name_aggregate
    ssa = load_per_name_aggregate()
    X_test = np.array(
        [encode_name(n) for n in ssa["name"].astype(str).str.lower()],
        dtype=np.int64,
    )
    y_test = (ssa["p_female"].to_numpy() >= 0.5).astype(np.int64)

    model.eval()
    all_p = []
    with torch.no_grad():
        for i in range(0, len(X_test), 1024):
            chunk = torch.from_numpy(X_test[i:i+1024])
            logits = model(chunk)
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    p_pred = np.concatenate(all_p)
    y_pred = (p_pred >= 0.5).astype(np.int64)
    acc = float((y_pred == y_test).mean())

    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_test, p_pred))
    except ValueError:
        auc = None

    print(f"  n_test = {len(X_test)}")
    print(f"  accuracy = {acc:.4f}")
    print(f"  ROC-AUC  = {auc:.4f}")
    print()
    print(f"  Published SOTA: Hu 2021 char-LSTM = 0.940 acc / 0.980 AUC")
    print(f"  v2 beat SOTA?   {'YES' if acc > 0.940 else 'NO'}")

    meta = {
        "seed":                 SEED,
        "max_len":              MAX_LEN,
        "n_params":             int(n_params),
        "embed_dim":            EMBED_DIM,
        "hidden_dim":           HIDDEN_DIM,
        "num_layers":           NUM_LAYERS,
        "dropout":              DROPOUT,
        "best_val_accuracy":    round(best_val_acc, 4),
        "ssa_test_accuracy":    round(acc, 4),
        "ssa_test_auc":         None if auc is None else round(auc, 4),
        "ssa_test_n":           int(len(X_test)),
        "published_sota_acc":   0.940,
        "published_sota_source":"Hu 2021, arXiv:2102.03692, Table 6",
        "beats_published_sota": bool(acc > 0.940),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Wrote {META_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
