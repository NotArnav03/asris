"""
FAIMR — Name training-corpus builder.

Reads two inputs:

  1. firstnames_raw.csv — upstream multi-cultural first-name database
     (GFDL 1.2+; see ATTRIBUTION.md and LICENSE-firstname-database.txt).
     ~46k names with a gender label (M, F, ?M, ?F, ?, =) and per-country
     usage frequencies on a -16..+16 scale.

  2. fairness/bias_detector.py — the in-repo curated seed lists
     (GENDERED_NAMES + _UNISEX_NAMES) used by the legacy detector.

Produces training_corpus.csv with the schema:

    name           : lower-cased first name (str)
    p_female       : empirical probability the name is used for female
                     individuals (float in [0, 1])
    p_male         : 1 - p_female (float in [0, 1])
    weight         : training weight (float).  Higher values mean we
                     trust this row more.  We assign:
                        - hard upstream labels  (M, F)   : 1.0
                        - soft upstream labels  (?M, ?F) : 0.7
                        - unisex upstream      (?, =)   : 0.5
                        - FAIMR seed (gendered)         : 0.6
                        - FAIMR seed (unisex)           : 0.5
                     Upstream wins on hard labels because it is sourced
                     from multi-country statistical aggregation; the
                     FAIMR seed is a curator's hand-list.
    culture        : best-guess culture cluster derived from the highest
                     per-country frequency (str).  One of:
                        western, south_asian, east_asian, arab, slavic,
                        european_other, other.  "other" is used when
                     the upstream row has no country information.
    source         : provenance tag.  One of:
                        firstname-db, faimr-seed, faimr-unisex,
                     possibly combined as e.g. "firstname-db+faimr-seed"
                     when both sources agree on the name.

The script is deterministic — given the same inputs it always emits the
same output.  It is meant to be re-run whenever firstnames_raw.csv or
the in-repo seed lists change.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fairness.bias_detector import GENDERED_NAMES, _UNISEX_NAMES  # noqa: E402


RAW_CSV = ROOT / "data" / "names" / "firstnames_raw.csv"
OUT_CSV = ROOT / "data" / "names" / "training_corpus.csv"


# Upstream gender label -> (p_female, label_weight)
_LABEL_MAP: dict[str, tuple[float, float]] = {
    "M":  (0.00, 1.0),
    "F":  (1.00, 1.0),
    "?M": (0.25, 0.7),
    "?F": (0.75, 0.7),
    "?":  (0.50, 0.5),
    "=":  (0.50, 0.5),
    # Some rows in the upstream file use "1M" (mainly male) or "1F"
    # (mainly female) variants; treat them as the soft labels.
    "1M": (0.25, 0.7),
    "1F": (0.75, 0.7),
}


# Country-column groupings.  These map the upstream country headers to
# a coarse culture cluster used downstream for stratified evaluation.
# Country names match the upstream header strings exactly.
_CULTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "western": (
        "Great Britain", "Ireland", "U.S.A.", "Australia", "Canada",
        "New Zealand",
    ),
    "european_other": (
        "Italy", "Malta", "Portugal", "Spain", "France", "Belgium",
        "Luxembourg", "the Netherlands", "East Frisia", "Germany",
        "Austria", "Swiss", "Iceland", "Denmark", "Norway", "Sweden",
        "Finland", "Estonia", "Latvia", "Lithuania", "Poland",
        "Czech Republic", "Slovakia", "Hungary", "Romania", "Bulgaria",
        "Greece",
    ),
    "slavic": (
        "Russia", "Belarus", "Moldova", "Ukraine", "Bosnia and Herzegovina",
        "Croatia", "Kosovo", "Macedonia", "Montenegro", "Serbia",
        "Slovenia", "Albania",
    ),
    "arab": (
        "Arabia/Persia", "Turkey", "Israel",
        "Armenia", "Azerbaijan", "Georgia",
        "Kazakhstan/Uzbekistan,etc.",
    ),
    "south_asian": (
        "India/Sri Lanka",
    ),
    "east_asian": (
        "China", "Japan", "Korea", "Vietnam",
    ),
}


def _country_value(cell: str) -> int:
    """Parse the upstream per-country frequency cell.

    Cells are either empty (no usage) or a signed integer in the range
    -16..+16.  We treat any non-empty value as "present" and use its
    absolute magnitude as a usage weight when picking the dominant
    culture for a name.
    """
    cell = (cell or "").strip()
    if not cell:
        return 0
    try:
        return abs(int(cell))
    except ValueError:
        return 0


def _classify_culture(row: dict) -> str:
    """Pick the culture cluster with the strongest per-country usage
    signal for this name.  Falls back to "other" when no country
    column has data."""
    best_group = "other"
    best_score = 0
    for group, countries in _CULTURE_GROUPS.items():
        score = sum(_country_value(row.get(c, "")) for c in countries)
        if score > best_score:
            best_score = score
            best_group = group
    return best_group


def _normalise(name: str) -> str:
    """Lower-case and strip non-letter characters from a name token."""
    return "".join(ch for ch in name.lower() if ch.isalpha())


def load_upstream() -> dict[str, dict]:
    """Read firstnames_raw.csv and return {name: row_dict}.

    Multiple upstream rows for the same normalised name are merged by
    averaging p_female (weighted by label_weight) and summing the
    culture-vote scores.  This handles upstream rows that disagree on
    gender (very rare) and rows that differ only in diacritics.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RAW_CSV}. Re-download from upstream — see ATTRIBUTION.md"
        )

    # accumulator[name] = {p_female_sum, weight_sum, culture_votes, raw_rows}
    accumulator: dict[str, dict] = defaultdict(
        lambda: {"p_female_w": 0.0, "weight_sum": 0.0,
                 "cultures": defaultdict(int), "rows": 0}
    )

    with RAW_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            raw_name = (row.get("name") or "").strip()
            name = _normalise(raw_name)
            if not name or len(name) < 2:
                continue
            label = (row.get("gender") or "").strip()
            if label not in _LABEL_MAP:
                continue
            p_f, w = _LABEL_MAP[label]
            acc = accumulator[name]
            acc["p_female_w"] += p_f * w
            acc["weight_sum"] += w
            acc["rows"] += 1
            # Culture: pick the highest-scoring group on THIS row, then
            # vote.  The dominant culture is the highest-voted group
            # across all rows for this name.
            culture = _classify_culture(row)
            acc["cultures"][culture] += 1

    out: dict[str, dict] = {}
    for name, acc in accumulator.items():
        if acc["weight_sum"] == 0:
            continue
        p_female = acc["p_female_w"] / acc["weight_sum"]
        culture = max(acc["cultures"].items(), key=lambda kv: kv[1])[0]
        out[name] = {
            "p_female": p_female,
            "weight": acc["weight_sum"] / acc["rows"],  # average label weight
            "culture": culture,
            "source": "firstname-db",
        }
    return out


def load_faimr_seed() -> dict[str, dict]:
    """Read the in-repo curated lists from bias_detector and convert
    them to the same schema.  Used both for new names not present in
    the upstream file and to boost the weight of names that BOTH
    sources agree on."""
    out: dict[str, dict] = {}
    for name in GENDERED_NAMES["male"]:
        out[_normalise(name)] = {
            "p_female": 0.0,
            "weight":   0.6,
            "culture":  "other",  # culture refined by upstream merge
            "source":   "faimr-seed",
        }
    for name in GENDERED_NAMES["female"]:
        out[_normalise(name)] = {
            "p_female": 1.0,
            "weight":   0.6,
            "culture":  "other",
            "source":   "faimr-seed",
        }
    for name in _UNISEX_NAMES:
        out[_normalise(name)] = {
            "p_female": 0.5,
            "weight":   0.5,
            "culture":  "other",
            "source":   "faimr-unisex",
        }
    return out


def merge(upstream: dict[str, dict], seed: dict[str, dict]) -> list[dict]:
    """Merge the two sources.

    Policy:
      - If a name is in BOTH sources, use the upstream p_female (more
        statistically grounded), but add the seed weight to indicate
        the curator's agreement.  Combine source tags.
      - If a name is only in upstream, take it as-is.
      - If a name is only in the FAIMR seed, take it as-is.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for name, row in upstream.items():
        seen.add(name)
        merged = dict(row)
        if name in seed:
            seed_row = seed[name]
            merged["weight"] = row["weight"] + seed_row["weight"]
            merged["source"] = f"firstname-db+{seed_row['source']}"
            # Culture stays upstream's — it has actual country data.
        rows.append({"name": name, **merged})
    for name, row in seed.items():
        if name in seen:
            continue
        rows.append({"name": name, **row})
    rows.sort(key=lambda r: r["name"])
    return rows


def write_corpus(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "p_female", "p_male", "weight", "culture", "source"],
        )
        writer.writeheader()
        for r in rows:
            p_f = round(r["p_female"], 4)
            writer.writerow({
                "name":     r["name"],
                "p_female": p_f,
                "p_male":   round(1.0 - p_f, 4),
                "weight":   round(r["weight"], 4),
                "culture":  r["culture"],
                "source":   r["source"],
            })


def _summarise(rows: list[dict]) -> None:
    """Print a coverage summary to stdout."""
    total = len(rows)
    culture_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    p_female_buckets = {"male<.1": 0, "leaning_m": 0, "unisex": 0,
                        "leaning_f": 0, "female>.9": 0}
    for r in rows:
        culture_counts[r["culture"]] += 1
        source_counts[r["source"]] += 1
        p = r["p_female"]
        if p < 0.10:
            p_female_buckets["male<.1"] += 1
        elif p < 0.40:
            p_female_buckets["leaning_m"] += 1
        elif p <= 0.60:
            p_female_buckets["unisex"] += 1
        elif p <= 0.90:
            p_female_buckets["leaning_f"] += 1
        else:
            p_female_buckets["female>.9"] += 1

    print(f"\nWrote {total} rows to {OUT_CSV.relative_to(ROOT)}\n")
    print("By culture:")
    for k, v in sorted(culture_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:>6}")
    print("\nBy source:")
    for k, v in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v:>6}")
    print("\nBy p_female bucket:")
    for k, v in p_female_buckets.items():
        print(f"  {k:<12} {v:>6}")


def main() -> None:
    print(f"Reading upstream from {RAW_CSV.relative_to(ROOT)} ...")
    upstream = load_upstream()
    print(f"  {len(upstream)} unique upstream names")

    print("Reading FAIMR seed from bias_detector.GENDERED_NAMES / _UNISEX_NAMES ...")
    seed = load_faimr_seed()
    print(f"  {len(seed)} seed names")

    rows = merge(upstream, seed)
    write_corpus(rows)
    _summarise(rows)


if __name__ == "__main__":
    main()
