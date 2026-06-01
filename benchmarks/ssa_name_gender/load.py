"""
FAIMR — SSA name-gender benchmark: dataset loader.

Loads the US Social Security Administration national baby-name
counts via the canonical Hadley Wickham mirror (the official SSA
zip at ssa.gov/oact/babynames blocks programmatic User-Agents,
so we use the well-known re-host).

Schema:
    year   int     1880..2008 in the Hadley snapshot
    name   str     given name (Title-cased)
    percent float  share of births of that sex in that year
    sex    str     "boy" or "girl"

We collapse the per-year frequencies into a per-name aggregate:

    name -> {male_births, female_births, p_female}

This is the input format FAIMR's name classifier expects.

Citation:
    United States Social Security Administration, National Baby
    Names Data, accessed via hadley/data-baby-names mirror.
    https://www.ssa.gov/oact/babynames/limits.html
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

SSA_URL = (
    "https://raw.githubusercontent.com/hadley/data-baby-names/master/"
    "baby-names.csv"
)
SSA_LOCAL = CACHE_DIR / "ssa_baby_names.csv"


def fetch(force: bool = False) -> Path:
    if SSA_LOCAL.exists() and not force:
        return SSA_LOCAL
    req = urllib.request.Request(SSA_URL, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {SSA_URL} -> {SSA_LOCAL.relative_to(ROOT)} ...")
    with urllib.request.urlopen(req, timeout=60) as resp, SSA_LOCAL.open("wb") as out:
        out.write(resp.read())
    return SSA_LOCAL


def load_per_name_aggregate(min_year: int = 1880, max_year: int = 2008):
    """Return a pandas DataFrame with one row per name and the
    aggregated male / female birth-share across [min_year, max_year].

    Columns:
        name         (str)
        male_share   (float, sum of percents in years with sex=boy)
        female_share (float, sum of percents in years with sex=girl)
        p_female     (float in [0, 1])
        gender_label (str: "male" if p_female <= 0.5 else "female")
        n_years      (int)
    """
    import pandas as pd
    path = fetch()
    df = pd.read_csv(path)
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].copy()
    df["name"] = df["name"].astype(str)

    # Pivot: per-name (male_share, female_share)
    g = df.groupby(["name", "sex"])["percent"].sum().unstack(fill_value=0.0)
    g.columns = [c if c != "boy" else "male_share" for c in g.columns]
    g.columns = [c if c != "girl" else "female_share" for c in g.columns]
    for col in ("male_share", "female_share"):
        if col not in g.columns:
            g[col] = 0.0
    g["total"] = g["male_share"] + g["female_share"]
    g = g[g["total"] > 0].copy()
    g["p_female"] = g["female_share"] / g["total"]
    g["gender_label"] = g["p_female"].apply(
        lambda p: "female" if p >= 0.5 else "male"
    )

    # Count years the name appeared (proxy for attestation strength)
    years = df.groupby("name")["year"].nunique().rename("n_years")
    g = g.join(years, how="left")

    return g.reset_index().drop(columns=["total"])


def main() -> int:
    df = load_per_name_aggregate()
    print(f"SSA baby names ({df['name'].nunique()} unique names)")
    print(f"  gender distribution: "
          f"{df['gender_label'].value_counts().to_dict()}")
    print(f"  most-attested name (n_years):")
    top = df.nlargest(5, "n_years")[["name", "n_years", "p_female"]]
    print(top.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
