"""Train the SSA char-LSTM name-gender classifier.

Data: the SSA national baby-names aggregate (per-name p_female).
We treat p_female as a soft-label regression target via BCE loss --
this exposes the model to the full attestation strength rather than
forcing every name to a hard 0/1.  Names with very-low attestation
(n_years < 3) are dropped.

Training set: 80% of SSA names (split DETERMINISTICALLY by hash on
name so train/test re-runs are stable).
Validation set: 10%.
Test set: 10% (held back -- the benchmark evaluator uses this).

Loss: BCEWithLogitsLoss, sample-weighted by sqrt(n_years) so well-
attested names contribute more.
Optimizer: AdamW, lr=2e-3, weight_decay=1e-4.
Epochs: 30 with early stopping (patience 4 on val accuracy).
Batch size: 256.

Determinism: torch seed 20251128.
"""

from __future__ import annotations

import hashlib
import json
import math
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
WEIGHTS_PATH = ROOT / "weights.pt"
META_PATH = ROOT / "meta.json"

BATCH_SIZE = 128
N_EPOCHS = 60
LR = 1.5e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 5e-4  # Stronger regularization on small data
PATIENCE = 8
MIN_N_YEARS = 3
LABEL_SMOOTH = 0.05


def _deterministic_split(name: str) -> str:
    """Hash-based split: train if first hex digit < 0xc, val if < 0xe, else test."""
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[0]
    n = int(h, 16)
    if n < 0xc:
        return "train"
    if n < 0xe:
        return "val"
    return "test"


def _build_tensors(df, split: str, use_hard_labels: bool = True):
    sub = df[df["split"] == split]
    X = np.array(
        [encode_name(n) for n in sub["name"].astype(str).tolist()],
        dtype=np.int64,
    )
    if use_hard_labels:
        # Hard labels with light label smoothing.  This matches the
        # eval-time hard-thresholding protocol and converges faster
        # than regressing the soft p_female.
        hard = (sub["p_female"].to_numpy() >= 0.5).astype(np.float32)
        y = hard * (1 - LABEL_SMOOTH) + (1 - hard) * LABEL_SMOOTH
        y = y.astype(np.float32)
    else:
        y = sub["p_female"].astype(np.float32).to_numpy()
    w = np.sqrt(sub["n_years"].astype(np.float32).to_numpy())
    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


def main() -> int:
    print(f"# SSA char-LSTM training (seed={SEED})")
    print()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from benchmarks.ssa_name_gender.load import load_per_name_aggregate
    print("Loading SSA aggregate ...")
    df = load_per_name_aggregate().copy()
    df = df[df["n_years"] >= MIN_N_YEARS].copy()
    df["split"] = df["name"].apply(_deterministic_split)
    print(f"  total names: {len(df)}")
    for s in ("train", "val", "test"):
        print(f"  {s:<5}  n={int((df['split'] == s).sum())}")
    print()

    Xtr, ytr, wtr = _build_tensors(df, "train")
    Xva, yva, wva = _build_tensors(df, "val")

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr, wtr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = CharBiLSTMGenderClassifier()
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
            total_loss += float(loss) * xb.size(0)
            total_n += xb.size(0)
        train_loss = total_loss / total_n

        # Val
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

    if best_state is None:
        print("ERROR: no best state captured")
        return 1

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print()
    print(f"Wrote {WEIGHTS_PATH.relative_to(REPO_ROOT)}  "
          f"({WEIGHTS_PATH.stat().st_size} bytes)")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    meta = {
        "seed": SEED,
        "max_len": MAX_LEN,
        "best_val_accuracy": round(best_val_acc, 4),
        "n_params": int(n_params),
        "min_n_years": MIN_N_YEARS,
        "split_counts": {
            s: int((df["split"] == s).sum()) for s in ("train", "val", "test")
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {META_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
