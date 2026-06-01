"""
FAIMR -- Fit the SSA second-stage recalibrator.

Why this script exists
======================

FAIMR's name-gender classifier uses per-culture isotonic calibration
fit on the upstream firstname-database distribution.  That distribution
is multicultural; the residual miscalibration on English-dominant rare
US names is the largest source of OOD error in the SSA-name-gender
benchmark (see benchmarks/ssa_name_gender/README.md, "Honest finding").

This script fits a SECOND-stage isotonic per culture cluster on the
SSA national baby-names corpus, then attaches it to the model pickle
under the field ``ssa_recalibrators``.

Fit set (LEAKAGE-FREE for the benchmark)
========================================

We fit ONLY on names that:
  * appear in the FAIMR training corpus (training_corpus.csv), AND
  * also appear in SSA aggregate -- so we have both a model
    prediction AND an SSA empirical p_female.

These overlap names are by construction DISJOINT from the OOD
holdout that benchmarks/ssa_name_gender/evaluate.py reports on, so
fitting the recalibrator here cannot leak into the headline Stage B
number.

Recalibrators are only fit for culture clusters where SSA is the
natural distribution (western, european_other, slavic) -- fitting
on tiny overlap for arab / south_asian / east_asian would do more
harm than good.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "fairness" / "names" / "model.pkl"
MODEL_CARD_PATH = REPO_ROOT / "fairness" / "names" / "model_card.json"
TRAINING_CORPUS_PATH = REPO_ROOT / "data" / "names" / "training_corpus.csv"

# Culture clusters where SSA is the natural distribution.  Recalibrators
# are NOT fit for arab / south_asian / east_asian -- the SSA overlap
# for those clusters is too small (mostly transliterated rare names)
# and the existing per-culture isotonic already calibrates them well.
SSA_DOMINANT_CULTURES = ("western", "european_other", "slavic")

# Minimum samples required to fit an isotonic per culture.  Below this,
# we skip the cluster (better to fall back to the existing per-culture
# isotonic than to fit on noise).
MIN_SAMPLES_PER_CULTURE = 200


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    from benchmarks.ssa_name_gender.load import load_per_name_aggregate
    from fairness.names.classifier import NameGenderClassifier

    print("Loading SSA aggregate ...")
    ssa = load_per_name_aggregate()
    print(f"  {len(ssa)} unique SSA names")

    print("Loading FAIMR training corpus ...")
    train = pd.read_csv(
        TRAINING_CORPUS_PATH, keep_default_na=False, na_values=[""],
    )
    train["name"] = train["name"].astype(str).str.lower()
    train_names = set(train["name"])
    print(f"  {len(train_names)} training names")

    # Overlap = the fit set (DISJOINT from the OOD benchmark holdout).
    ssa["name_lc"] = ssa["name"].str.lower()
    overlap = ssa[ssa["name_lc"].isin(train_names)].copy()
    print(f"  {len(overlap)} overlap names available for recalibrator fit")
    print(f"  (these are DISJOINT from the benchmark OOD holdout)")
    print()

    # Run the existing classifier on the overlap names; we need both
    # the post-per-culture-isotonic probability AND the predicted
    # culture cluster.  Important: bypass the lookup fastpath -- we
    # want the MODEL prediction, since that's what the recalibrator
    # will be applied to in production.
    print("Loading FAIMR classifier ...")
    clf = NameGenderClassifier()
    clf._ensure_loaded()
    model = clf._model

    print("Scoring overlap names through the model (bypassing lookup) ...")
    t0 = time.time()
    names_lc = overlap["name_lc"].tolist()
    raw_post_iso = model.predict_proba(names_lc)[:, 1]
    cultures = model.predict_culture(names_lc)
    print(f"  scored {len(names_lc)} names in {time.time() - t0:.1f}s")
    print()

    overlap["model_p_female"] = raw_post_iso
    overlap["predicted_culture"] = cultures

    print("Per-culture overlap distribution:")
    by_culture = overlap.groupby("predicted_culture").size()
    for c, n in by_culture.items():
        print(f"  {c:<20}  n={n}")
    print()

    # Fit per-culture isotonic.
    recalibrators: dict = {}
    print("Fitting per-culture SSA recalibrators ...")
    for culture in SSA_DOMINANT_CULTURES:
        slice_df = overlap[overlap["predicted_culture"] == culture]
        n = len(slice_df)
        if n < MIN_SAMPLES_PER_CULTURE:
            print(f"  {culture:<20}  SKIP (n={n} < {MIN_SAMPLES_PER_CULTURE})")
            continue
        x = slice_df["model_p_female"].to_numpy()
        y = slice_df["p_female"].to_numpy()  # SSA empirical p_female
        iso = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        )
        iso.fit(x, y)
        # Quick sanity: pre vs post mean abs deviation from SSA truth.
        pre = float(np.mean(np.abs(x - y)))
        post = float(np.mean(np.abs(iso.predict(x) - y)))
        improvement = pre - post
        verdict = "KEEP" if improvement > 0.005 else "DROP"
        print(f"  {culture:<20}  n={n}  pre-MAE={pre:.4f}  post-MAE={post:.4f}  "
              f"(reduction {(1 - post/pre)*100:+.1f}%)  {verdict}")
        if verdict == "KEEP":
            recalibrators[culture] = iso
    print()

    if not recalibrators:
        print("ERROR: no recalibrators fit -- aborting.")
        return 1

    # Attach to the model and re-pickle.
    model.ssa_recalibrators = recalibrators
    print(f"Writing updated model to {MODEL_PATH.relative_to(REPO_ROOT)} ...")
    backup = MODEL_PATH.with_suffix(".pkl.bak")
    if MODEL_PATH.exists() and not backup.exists():
        backup.write_bytes(MODEL_PATH.read_bytes())
        print(f"  backup -> {backup.relative_to(REPO_ROOT)}")
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    new_sha = _sha256(MODEL_PATH)
    new_size = MODEL_PATH.stat().st_size
    print(f"  new SHA-256: {new_sha}")
    print(f"  new size:    {new_size} bytes")
    print()

    # Update model_card.json to track the new artefact.
    print(f"Updating {MODEL_CARD_PATH.relative_to(REPO_ROOT)} ...")
    card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    card.setdefault("integrity", {})["sha256"] = new_sha
    card["integrity"]["size_bytes"] = new_size
    card.setdefault("pipeline", {})["ssa_recalibration"] = {
        "enabled": True,
        "cultures": sorted(recalibrators.keys()),
        "fit_set":  "training_corpus.csv ∩ SSA aggregate",
        "fit_set_n": int(len(overlap)),
        "fit_set_disjoint_from_benchmark_holdout": True,
    }
    MODEL_CARD_PATH.write_text(
        json.dumps(card, indent=2), encoding="utf-8",
    )
    print(f"  card updated, version preserved.")
    print()
    print("Done.  Re-run benchmarks/ssa_name_gender/evaluate.py to verify "
          "Stage B improvement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
