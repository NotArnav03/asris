"""
FAIMR -- TREC Fair Ranking 2022 benchmark: dataset loader.

Uses the TREC 2022 Fair Ranking Track training split (50 queries
representing Wikimedia WikiProjects, ~2.1M qrels).  We do NOT need
the full Wikipedia plain-text corpus -- only the per-document
"articles_discrete" file with the fairness attribute metadata.

Citations:
  Ekstrand et al., "Overview of the TREC 2022 Fair Ranking Track",
  arXiv:2302.05558.

Downloads:
  - trec_2022_train_reldocs.jsonl  (18 MB; per-query rel-doc lists)
  - trec_2022_articles_discrete.json.gz  (237 MB; per-doc metadata)

Both come from:
  https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/2022/

We rely on ir_datasets to do the initial download (it caches the
files under ~/.ir_datasets/trec-fair/2022/), but we read them
directly because ir_datasets's iteration over the full corpus is
prohibitively slow for our purposes.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

CACHE = Path.home() / ".ir_datasets" / "trec-fair" / "2022"
RELDOCS = CACHE / "trec_2022_train_reldocs.jsonl"
ARTICLES = CACHE / "trec_2022_articles_discrete.json.gz"


def ensure_data() -> None:
    """Download the two needed files via ir_datasets if missing.

    We trigger ir_datasets's download by calling queries_iter once;
    that materialises the reldocs.jsonl.  The articles_discrete
    file is downloaded by docs_iter -- we catch the in-progress
    state by checking if both files exist after one queries call.
    """
    if RELDOCS.exists() and ARTICLES.exists():
        return
    print("Downloading TREC Fair Ranking 2022 train via ir_datasets ...")
    import ir_datasets
    ds = ir_datasets.load("trec-fair/2022/train")
    # docs_count() ensures the articles_discrete file is downloaded
    # without iterating individual docs (which is slow).
    _ = list(ds.queries_iter())
    # Force the docs metadata file (articles_discrete) to download.
    # Touch the docs_handler via a method that does not iterate.
    if not ARTICLES.exists():
        raise RuntimeError(
            f"ir_datasets did not produce {ARTICLES}.  "
            "Download manually from "
            "https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/2022/"
        )


def load_queries() -> list[dict]:
    """Return [{"query_id": ..., "title": ..., "rel_docs": [...]}, ...]
    for the 50 train queries."""
    ensure_data()
    queries: list[dict] = []
    with RELDOCS.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries.append({
                "query_id": int(d["id"]),
                "title":    str(d["title"]),
                "rel_docs": [int(x) for x in d["rel_docs"]],
            })
    return queries


def iter_articles_discrete() -> Iterator[dict]:
    """Yield {page_id, gender, gender_category, pred_qual, qual_cat,
    page_subcont_regions, relative_pageviews_category, ...} for every
    doc.  Used to build a doc_id -> attribute lookup."""
    ensure_data()
    with gzip.open(ARTICLES, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_doc_lookup(
    needed_ids: set[int] | None = None,
) -> dict[int, dict]:
    """Return doc_id -> attribute dict.  If `needed_ids` is given,
    only those docs are kept (much smaller in-memory footprint)."""
    out: dict[int, dict] = {}
    for d in iter_articles_discrete():
        page_id = int(d["page_id"])
        if needed_ids is not None and page_id not in needed_ids:
            continue
        out[page_id] = d
    return out


def main() -> int:
    queries = load_queries()
    print(f"Queries: {len(queries)}")
    for q in queries[:5]:
        print(f"  {q['query_id']:>4}  {q['title']:<30}  "
              f"n_rel={len(q['rel_docs'])}")
    print(f"  ...")
    total_rel = sum(len(q["rel_docs"]) for q in queries)
    print(f"Total rel doc-query pairs: {total_rel}")

    needed = {d for q in queries for d in q["rel_docs"]}
    print(f"Unique rel docs across queries: {len(needed)}")

    print("Loading metadata for relevant docs only ...")
    docs = build_doc_lookup(needed_ids=needed)
    print(f"Loaded metadata for {len(docs)} relevant docs")

    n_with_gender = sum(1 for d in docs.values() if d.get("gender"))
    print(f"  with non-empty gender list: {n_with_gender}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
