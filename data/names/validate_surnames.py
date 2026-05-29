"""
FAIMR — Surname denylist coverage validator.

Measures the fraction of common surnames in each cultural cluster
that the denylist (data/names/surnames.csv) actually catches.

Holdout source: data/names/surname_holdout.csv — a manually curated
list of well-attested common surnames per cluster, with the
authoritative public source recorded inline.  Holdout is small
(~90 surnames) but specifically chosen to probe coverage at the
"any reasonable resume would include this name" level.

Output:
  - Prints per-culture coverage to stdout.
  - Updates fairness/names/model_card.json under
    surname_coverage = {culture: {n, hits, coverage}, ...}.

A regression test (TestSurnameCoverage) asserts the per-culture
coverage stays above _COVERAGE_FLOOR (default 0.70).  When a future
edit drops below the floor the test fires immediately — so coverage
degradation is caught before shipping.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DENYLIST_CSV = ROOT / "data" / "names" / "surnames.csv"
HOLDOUT_CSV = ROOT / "data" / "names" / "surname_holdout.csv"
MODEL_CARD = ROOT / "fairness" / "names" / "model_card.json"


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalpha())


def load_denylist() -> set:
    out: set = set()
    if not DENYLIST_CSV.exists():
        raise FileNotFoundError(
            f"{DENYLIST_CSV} missing — run data/names/build_surnames.py first"
        )
    with DENYLIST_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tok = _normalise(row.get("name", ""))
            if tok:
                out.add(tok)
    return out


def load_holdout() -> dict:
    if not HOLDOUT_CSV.exists():
        raise FileNotFoundError(f"{HOLDOUT_CSV} missing")
    holdout: dict = defaultdict(list)
    with HOLDOUT_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tok = _normalise(row.get("name", ""))
            culture = row.get("culture", "unknown")
            if tok:
                holdout[culture].append(tok)
    return holdout


def measure_coverage(denylist: set, holdout: dict) -> dict:
    coverage: dict = {}
    for culture, surnames in holdout.items():
        hits = [s for s in surnames if s in denylist]
        misses = [s for s in surnames if s not in denylist]
        coverage[culture] = {
            "n":         len(surnames),
            "hits":      len(hits),
            "misses":    misses,                       # surface for debugging
            "coverage":  round(len(hits) / len(surnames), 4),
        }
    return coverage


def update_model_card(coverage: dict) -> None:
    if not MODEL_CARD.exists():
        print(f"model card not found at {MODEL_CARD}; skipping update")
        return
    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    # We strip the verbose `misses` field from the model card to keep
    # the file readable; tests can still load the holdout and recompute.
    card["surname_coverage"] = {
        culture: {k: v for k, v in stats.items() if k != "misses"}
        for culture, stats in coverage.items()
    }
    MODEL_CARD.write_text(
        json.dumps(card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    denylist = load_denylist()
    holdout = load_holdout()
    coverage = measure_coverage(denylist, holdout)

    print(f"\nLoaded {len(denylist)} surnames from denylist.")
    total_in = sum(s["n"] for s in coverage.values())
    total_hits = sum(s["hits"] for s in coverage.values())
    print(f"Holdout: {total_in} surnames across {len(coverage)} cultures.")
    print(f"Overall coverage: {total_hits}/{total_in} = "
          f"{total_hits / total_in:.1%}\n")
    for culture, stats in sorted(coverage.items()):
        print(f"  {culture:<16} {stats['hits']:>3}/{stats['n']:<3}  "
              f"coverage={stats['coverage']:.1%}")
        if stats["misses"]:
            print(f"    misses: {', '.join(stats['misses'])}")

    update_model_card(coverage)
    print(f"\nUpdated {MODEL_CARD.relative_to(ROOT)}.")


if __name__ == "__main__":
    sys.exit(main() or 0)
