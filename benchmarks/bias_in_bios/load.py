"""
FAIMR — Bias in Bios benchmark: dataset loader.

Downloads the Bias in Bios test split from HuggingFace
(LabHC/bias_in_bios) — the canonical re-host of the De-Arteaga
et al. 2019 dataset.  The original Microsoft repo is archived;
LabHC maintains the parquet conversion with the same labels.

Schema:
    hard_text  str   — biography with the candidate's name redacted
                       to a placeholder.  Pronouns (he/she/his/her)
                       are kept and are the primary gender signal.
    profession int   — 0..27, one of 28 occupations
                       (mapped via PROFESSION_LABELS below).
    gender     int   — 0 = male, 1 = female (per dataset convention;
                       verified empirically by pronoun frequency).

Citation:
    Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer
    Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik,
    Krishnaram Kenthapadi, Adam Tauman Kalai. "Bias in Bios: A
    Case Study of Semantic Representation Bias in a High-Stakes
    Setting." FAccT 2019.

License: CC0 (HuggingFace re-host); original Common Crawl scrape
under the Common Crawl terms of use.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

TEST_URL = (
    "https://huggingface.co/datasets/LabHC/bias_in_bios/resolve/main/"
    "data/test-00000-of-00001-5598c840ce8de1ee.parquet"
)
TEST_LOCAL = CACHE_DIR / "test.parquet"

# Occupation labels (De-Arteaga 2019 Table 1)
PROFESSION_LABELS: dict = {
    0:  "accountant",
    1:  "architect",
    2:  "attorney",
    3:  "chiropractor",
    4:  "comedian",
    5:  "composer",
    6:  "dentist",
    7:  "dietitian",
    8:  "dj",
    9:  "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher",
}

GENDER_LABELS: dict = {0: "male", 1: "female"}


def fetch_test_split(force: bool = False) -> Path:
    """Download the test parquet to ``_cache/test.parquet``.
    Idempotent — re-uses the cached file unless ``force=True``."""
    if TEST_LOCAL.exists() and not force:
        return TEST_LOCAL
    print(f"Downloading {TEST_URL} -> {TEST_LOCAL.relative_to(ROOT)} ...")
    urllib.request.urlretrieve(TEST_URL, TEST_LOCAL)
    print(f"  cached {TEST_LOCAL.stat().st_size // 1024} KB")
    return TEST_LOCAL


def load_test():
    """Load the test split as a pandas DataFrame."""
    import pandas as pd
    path = fetch_test_split()
    df = pd.read_parquet(path)
    return df


def main() -> int:
    df = load_test()
    print(f"Bias in Bios test split: {len(df)} rows")
    print(f"  columns: {df.columns.tolist()}")
    print(f"  gender distribution: "
          f"{df['gender'].value_counts().to_dict()}")
    print(f"  profession distribution (top 5):")
    for prof_id, n in df['profession'].value_counts().head(5).items():
        print(f"    {PROFESSION_LABELS[prof_id]:<20} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
