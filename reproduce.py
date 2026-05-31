"""
FAIMR — Reproducibility runner.

Single entry point that rebuilds every derived artefact in the
project from its committed inputs, in dependency order:

    1.  data/names/firstnames_raw.csv         (committed upstream)
        data/names/us_surnames_raw.csv        (committed upstream)
              |
              v
    2.  data/names/build_corpus.py            -> training_corpus.csv
    3.  data/names/build_surnames.py          -> surnames.csv
              |
              v
    4.  fairness/names/train_classifier.py    -> model.pkl + model_card.json
    5.  data/names/validate_surnames.py       -> surname coverage in card
              |
              v
    6.  pytest -k "TestBiasDetector or ..."   (must pass)

Each step prints a banner with the stage name and the SHA-256 of the
artefact it produced.  The runner also emits a top-level
``reproducibility_manifest.json`` with:

  - timestamps for each stage
  - SHA-256 of every committed and derived artefact
  - the Python + sklearn + scipy versions used
  - the random seeds in effect
  - pass/fail of each stage

A reviewer can run ``python reproduce.py`` from a fresh checkout
and verify that the artefacts they obtain match the values pinned
in the manifest under git.

Usage::

    python reproduce.py                       # full rebuild + tests
    python reproduce.py --skip-train          # use the committed model
    python reproduce.py --grid-search         # also re-search hyperparams
    python reproduce.py --skip-tests          # skip the pytest step
    python reproduce.py --check               # verify hashes match
                                              # (no rebuild — read-only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "reproducibility_manifest.json"

# Files whose hashes we record on every run.  Committed inputs are
# fingerprinted so the manifest can prove WHICH version of the upstream
# data this reproduction was derived from; derived outputs are
# fingerprinted so the reviewer can cross-check artefacts byte-for-byte.
COMMITTED_INPUTS: list = [
    "data/names/firstnames_raw.csv",
    "data/names/us_surnames_raw.csv",
    "data/names/surname_holdout.csv",
    "data/names/nicknames.csv",
    "fairness/names/seed_lists.py",
    "fairness/names/train_classifier.py",
    "fairness/names/cultural_classifier.py",
    "data/names/build_corpus.py",
    "data/names/build_surnames.py",
    "data/names/validate_surnames.py",
]
DERIVED_OUTPUTS: list = [
    "data/names/training_corpus.csv",
    "data/names/surnames.csv",
    "fairness/names/model.pkl",
    "fairness/names/model_card.json",
]


def _sha256(path: Path) -> str:
    """Stream-hash a file; tolerate missing paths."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _banner(text: str) -> None:
    bar = "=" * 70
    print()
    print(bar)
    print(f"  {text}")
    print(bar)


def _run(cmd: list, label: str) -> dict:
    """Run a subprocess and capture the timing + exit code."""
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    seconds = time.time() - t0
    ok = proc.returncode == 0
    print(f"  {label} {'OK' if ok else 'FAIL'} ({seconds:.1f}s)")
    return {
        "label":        label,
        "cmd":          cmd,
        "returncode":   proc.returncode,
        "ok":           ok,
        "seconds":      round(seconds, 1),
    }


def _collect_versions() -> dict:
    """Record the Python + key library versions in the manifest so a
    reviewer can spot environment drift between their reproduction and
    the original.  Best-effort — missing libs surface as None.
    Maps PyPI dist name -> import module name so we use the
    discoverable form in the manifest but read __version__ from the
    importable one.
    """
    pkgs: list = [
        # (dist_name, import_name)
        ("numpy",        "numpy"),
        ("pandas",       "pandas"),
        ("scikit-learn", "sklearn"),
        ("scipy",        "scipy"),
        ("fastapi",      "fastapi"),
        ("throttled-py", "throttled"),
    ]
    out: dict = {"python": sys.version.split()[0]}
    for dist, mod_name in pkgs:
        try:
            mod = __import__(mod_name)
            out[dist] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[dist] = None
    return out


def _collect_seeds() -> dict:
    """Record the random seeds used by the training pipeline.  Drift
    in any of these would produce a different model.pkl from
    byte-identical inputs."""
    try:
        sys.path.insert(0, str(ROOT))
        from fairness.names import train_classifier
        return {
            "train_classifier.RANDOM_STATE": train_classifier.RANDOM_STATE,
        }
    except Exception:
        return {}


def _hash_set(paths: list) -> dict:
    return {p: _sha256(ROOT / p) for p in paths}


def reproduce(
    skip_train: bool = False,
    grid_search: bool = False,
    skip_tests: bool = False,
) -> dict:
    """Run the full pipeline and return the manifest dict."""
    manifest: dict = {
        "started_at":         datetime.now(timezone.utc).isoformat(),
        "versions":           _collect_versions(),
        "seeds":              _collect_seeds(),
        "input_hashes":       _hash_set(COMMITTED_INPUTS),
        "output_hashes_before": _hash_set(DERIVED_OUTPUTS),
        "stages":             [],
    }

    py = sys.executable

    _banner("Stage 1: build training corpus")
    manifest["stages"].append(_run(
        [py, "data/names/build_corpus.py"],
        "training_corpus",
    ))

    _banner("Stage 2: build surname denylist")
    manifest["stages"].append(_run(
        [py, "data/names/build_surnames.py"],
        "surnames",
    ))

    if not skip_train:
        _banner("Stage 3: train classifier")
        train_cmd = [py, "fairness/names/train_classifier.py"]
        if grid_search:
            train_cmd.append("--grid-search")
        manifest["stages"].append(_run(train_cmd, "train_classifier"))
    else:
        print("(stage 3 skipped — using committed model)")

    _banner("Stage 4: validate surname denylist coverage")
    manifest["stages"].append(_run(
        [py, "data/names/validate_surnames.py"],
        "validate_surnames",
    ))

    if not skip_tests:
        _banner("Stage 5: regression tests")
        manifest["stages"].append(_run(
            [py, "-m", "pytest", "tests/test_core.py",
             "-q", "--tb=short",
             "-k", "TestBiasDetector or TestNameClassifier or "
                   "TestSurnameCoverage or TestCounterfactualRobustness or "
                   "TestConstrainedInsertionFCR or TestApiHardening or "
                   "TestFairnessRanker or TestEvaluationMetrics or "
                   "TestCounterfactual or TestTextNormalizer"],
            "pytest",
        ))
    else:
        print("(stage 5 skipped)")

    manifest["finished_at"]        = datetime.now(timezone.utc).isoformat()
    manifest["output_hashes_after"] = _hash_set(DERIVED_OUTPUTS)
    manifest["all_stages_ok"]      = all(s["ok"] for s in manifest["stages"])

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def check_only() -> int:
    """Verify that the committed manifest hashes match the current
    files on disk.  Use after a fresh checkout to confirm artefact
    integrity without re-running the pipeline."""
    if not MANIFEST_PATH.exists():
        print(f"No manifest at {MANIFEST_PATH.relative_to(ROOT)}; "
              f"run `python reproduce.py` first.")
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("output_hashes_after", {})
    mismatches: list = []
    for path, want in expected.items():
        got = _sha256(ROOT / path)
        if want != got:
            mismatches.append((path, want, got))
    if mismatches:
        print("Hash MISMATCH on:")
        for path, want, got in mismatches:
            print(f"  {path}")
            print(f"    want: {want}")
            print(f"    got : {got}")
        return 1
    print("All derived-artefact hashes match the manifest.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce FAIMR's derived artefacts from committed inputs."
    )
    ap.add_argument("--skip-train",  action="store_true",
                    help="reuse the committed model.pkl")
    ap.add_argument("--grid-search", action="store_true",
                    help="grid-search hyperparameters during training")
    ap.add_argument("--skip-tests",  action="store_true",
                    help="don't run pytest at the end")
    ap.add_argument("--check",       action="store_true",
                    help="check artefact hashes against manifest "
                         "(no rebuild)")
    args = ap.parse_args()

    if args.check:
        return check_only()

    manifest = reproduce(
        skip_train=args.skip_train,
        grid_search=args.grid_search,
        skip_tests=args.skip_tests,
    )
    print()
    print(f"Manifest written to {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"All stages OK: {manifest['all_stages_ok']}")
    return 0 if manifest["all_stages_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
