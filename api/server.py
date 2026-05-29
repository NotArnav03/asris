"""
FAIMR — REST API Server
FastAPI-based API for resume-JD matching, ranking, and explanations.

Usage:
    python -m api.server
    # or: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    API_HOST, API_PORT, BASE_DIR, PROCESSED_RESUME_DIR, RAW_JD_DIR,
    LABELED_DIR, get_logger
)

logger = get_logger("api.server")

# ─── FastAPI App ─────────────────────────────────────────────────

# --- API hardening configuration -------------------------------------------
# All defaults can be overridden via environment variables.  Values are
# chosen to be safe for development; production should tighten the CORS
# allowlist and set FAIMR_API_KEY to require authenticated access.

# Per-resume body cap.  100 KB comfortably fits any real resume; values
# beyond this almost certainly indicate either an attack (DoS via giant
# pasted text) or a misconfigured upload.
MAX_RESUME_BYTES = int(os.getenv("FAIMR_MAX_RESUME_BYTES", "100000"))

# Max number of resumes in a single ranking / audit request.
MAX_REQUEST_RESUMES = int(os.getenv("FAIMR_MAX_REQUEST_RESUMES", "5000"))

# Total request body cap.  5000 resumes * ~100 KB worst case = 500 MB,
# but that's the absolute ceiling; the default below is much tighter.
MAX_TOTAL_REQUEST_BYTES = int(
    os.getenv("FAIMR_MAX_TOTAL_BYTES", str(50 * 1024 * 1024))  # 50 MB
)

# Token-bucket rate limit: requests per window per IP.
RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("FAIMR_RATE_LIMIT_WINDOW_SECONDS", "60")
)
RATE_LIMIT_REQUESTS = int(os.getenv("FAIMR_RATE_LIMIT_REQUESTS", "60"))

# Optional API key.  When set, every protected endpoint requires the
# header "X-API-Key: <value>".  Empty string (default) disables auth.
API_KEY = os.getenv("FAIMR_API_KEY", "").strip()

# CORS allowlist.  Comma-separated list of origins.  Default is the
# permissive "*"; production deployments should set this to a list of
# trusted origins.  A single "*" is still allowed for local dev.
_cors_env = os.getenv("FAIMR_CORS_ORIGINS", "*").strip()
ALLOWED_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]


app = FastAPI(
    title="FAIMR — Fairness-Aware Interpretable Multi-Signal Ranking API",
    description="Match resumes to job descriptions using multi-signal AI ranking.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False if "*" in ALLOWED_ORIGINS else True,
)


# --- Rate limiter middleware -----------------------------------------------
# Sliding window per remote address.  Kept in-process (a single fastapi
# worker); a production multi-worker deployment should use a shared
# Redis backend, but the in-process limiter is enough to neutralise the
# casual flood-attack vector documented in the security review.
_rate_buckets: dict = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = (request.client.host if request.client else "unknown")
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    # Drop expired entries
    while bucket and bucket[0] < now - RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_REQUESTS:
        return HTMLResponse(
            content=(
                f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS}s.  Retry shortly."
            ),
            status_code=429,
        )
    bucket.append(now)
    return await call_next(request)


def require_api_key(request: Request) -> None:
    """FastAPI dependency that checks the X-API-Key header when
    FAIMR_API_KEY is set in the environment.  No-op when unset."""
    if not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header",
        )


def _validate_resume_texts(resume_texts: dict) -> None:
    """Reject the request when the resume_texts payload violates the
    configured caps.  Raises HTTPException(413) on size violations,
    422 on schema violations."""
    if not isinstance(resume_texts, dict):
        raise HTTPException(
            status_code=422,
            detail="resume_texts must be a JSON object",
        )
    n = len(resume_texts)
    if n > MAX_REQUEST_RESUMES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many resumes: {n} > {MAX_REQUEST_RESUMES} cap. "
                f"Split the corpus into smaller batches."
            ),
        )
    total_bytes = 0
    for filename, text in resume_texts.items():
        if not isinstance(text, str):
            raise HTTPException(
                status_code=422,
                detail=f"resume_texts[{filename!r}] must be a string",
            )
        body_bytes = len(text.encode("utf-8"))
        if body_bytes > MAX_RESUME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Resume {filename!r} is {body_bytes} bytes; "
                    f"max is {MAX_RESUME_BYTES}.  Inspect the upload — "
                    f"a real resume rarely exceeds 100 KB."
                ),
            )
        total_bytes += body_bytes
        if total_bytes > MAX_TOTAL_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Total request body exceeds "
                    f"{MAX_TOTAL_REQUEST_BYTES} bytes.  "
                    f"Reduce the number of resumes per call."
                ),
            )

# Mount frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ─── Lazy-loaded Components ─────────────────────────────────────

_embedding_manager = None
_explainer = None

def get_embedding_manager():
    global _embedding_manager
    if _embedding_manager is None:
        from embeddings.embedding_manager import EmbeddingManager
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

def get_explainer():
    global _explainer
    if _explainer is None:
        from explainability.explainer import MatchExplainer
        _explainer = MatchExplainer()
    return _explainer


@app.on_event("startup")
async def startup_event():
    get_embedding_manager()


# ─── Request/Response Models ────────────────────────────────────

class RankRequest(BaseModel):
    jd_text: str
    resume_texts: dict[str, str]  # {filename: text}
    top_k: Optional[int] = 10

class RankResult(BaseModel):
    filename: str
    score: float
    rank: int

class RankResponse(BaseModel):
    ranked_candidates: list[RankResult]
    total_candidates: int

class ExplainRequest(BaseModel):
    jd_text: str
    resume_text: str
    job_id: Optional[str] = "api_query"

class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>FAIMR API is running. Frontend not found.</h1>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF resume and extract its text.
    Returns the extracted text for use in ranking/explaining.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    import pdfplumber
    import io

    try:
        contents = await file.read()
        text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")

        # Basic cleaning
        import re
        text = re.sub(r'\S+@\S+', '[email]', text)
        text = re.sub(r'\+?\d[\d -]{8,12}\d', '[phone]', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return {
            "filename": file.filename,
            "text": text,
            "pages": len(pdf.pages) if hasattr(pdf, 'pages') else 0,
            "word_count": len(text.split()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")


@app.post("/rank-pdfs")
async def rank_pdf_resumes(
    jd_text: str = File(...),
    files: list[UploadFile] = File(...),
    top_k: int = 10,
):
    """
    Upload multiple PDF resumes and rank them against a JD.
    """
    import pdfplumber
    import io

    resume_texts = {}
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        try:
            contents = await file.read()
            text = ""
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                resume_texts[file.filename] = text.strip()
        except Exception:
            continue

    if not resume_texts:
        raise HTTPException(status_code=400, detail="No valid PDF resumes found")

    manager = get_embedding_manager()
    jd_emb = manager.sbert_model.encode(jd_text, convert_to_numpy=True)
    resume_embs = manager.encode_sbert(resume_texts, use_cache=True)

    filenames = list(resume_embs.keys())
    resume_matrix = np.vstack(list(resume_embs.values()))
    jd_vec = jd_emb.reshape(1, -1)
    norms = np.linalg.norm(resume_matrix, axis=1, keepdims=True) * np.linalg.norm(jd_vec)
    sims = (resume_matrix @ jd_vec.T).flatten() / np.maximum(norms.flatten(), 1e-10)

    scored = sorted(zip(filenames, sims.tolist()), key=lambda x: x[1], reverse=True)

    results = [
        {"filename": f, "score": round(s, 4), "rank": i + 1}
        for i, (f, s) in enumerate(scored[:top_k])
    ]

    return {"ranked_candidates": results, "total_candidates": len(scored)}


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(
    request: RankRequest,
    _api: None = Depends(require_api_key),
):
    """
    Rank resumes against a job description.
    Accepts a JD and a dict of resumes, returns ranked candidates.
    """
    _validate_resume_texts(request.resume_texts)
    if not request.resume_texts:
        raise HTTPException(status_code=400, detail="No resume texts provided")

    manager = get_embedding_manager()

    jd_emb = manager.sbert_model.encode(request.jd_text, convert_to_numpy=True)
    resume_embs = manager.encode_sbert(request.resume_texts, use_cache=True)

    filenames = list(resume_embs.keys())
    resume_matrix = np.vstack(list(resume_embs.values()))
    jd_vec = jd_emb.reshape(1, -1)
    norms = np.linalg.norm(resume_matrix, axis=1, keepdims=True) * np.linalg.norm(jd_vec)
    sims = (resume_matrix @ jd_vec.T).flatten() / np.maximum(norms.flatten(), 1e-10)

    scored = sorted(zip(filenames, sims.tolist()), key=lambda x: x[1], reverse=True)

    results = [
        RankResult(filename=f, score=round(s, 4), rank=i + 1)
        for i, (f, s) in enumerate(scored[:request.top_k])
    ]

    return RankResponse(
        ranked_candidates=results,
        total_candidates=len(scored),
    )


@app.post("/explain")
async def explain_match(
    request: ExplainRequest,
    _api: None = Depends(require_api_key),
):
    """
    Explain why a resume matches (or doesn't match) a job description.
    Returns skill analysis, keyword overlap, and a human-readable verdict.
    """
    # Per-resume length cap also applies here (single-resume endpoint).
    if len(request.resume_text.encode("utf-8")) > MAX_RESUME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Resume body exceeds {MAX_RESUME_BYTES}-byte cap.",
        )
    explainer = get_explainer()
    manager = get_embedding_manager()

    # SBERT score
    jd_emb = manager.sbert_model.encode(request.jd_text, convert_to_numpy=True)
    resume_emb = manager.sbert_model.encode(request.resume_text, convert_to_numpy=True)
    sbert_score = float(manager.cosine_similarity(jd_emb, resume_emb))

    explanation = explainer.explain_match(
        job_id=request.job_id,
        jd_text=request.jd_text,
        resume_text=request.resume_text,
        sbert_score=sbert_score,
    )

    return explanation


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    resume_count = len(list(PROCESSED_RESUME_DIR.glob("*.txt")))

    jd_count = 0
    balanced_path = RAW_JD_DIR / "postings_balanced.csv"
    if balanced_path.exists():
        import pandas as pd
        jd_count = len(pd.read_csv(balanced_path))

    pair_files = list(LABELED_DIR.glob("*.csv"))
    pairs_info = {}
    for f in pair_files:
        import pandas as pd
        df = pd.read_csv(f)
        pairs_info[f.stem] = len(df)

    return {
        "resumes_processed": resume_count,
        "jds_balanced": jd_count,
        "pair_datasets": pairs_info,
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get embedding cache statistics."""
    manager = get_embedding_manager()
    return manager.cache_stats()


@app.post("/audit")
async def audit_fairness(
    request: RankRequest,
    _api: None = Depends(require_api_key),
):
    """
    Run a fairness audit on ranked candidates.
    Returns bias metrics + fairness-constrained re-ranking.
    """
    _validate_resume_texts(request.resume_texts)
    if not request.resume_texts:
        raise HTTPException(status_code=400, detail="No resume texts provided")

    manager = get_embedding_manager()
    jd_emb = manager.sbert_model.encode(request.jd_text, convert_to_numpy=True)
    resume_embs = manager.encode_sbert(request.resume_texts, use_cache=False)

    # Score candidates
    scores = {}
    for filename, emb in resume_embs.items():
        scores[filename] = float(manager.cosine_similarity(jd_emb, emb))

    # Bias audit
    from fairness.bias_detector import BiasDetector
    detector = BiasDetector()
    audit = detector.audit_ranking_bias(request.resume_texts, scores)

    # Fairness-constrained re-ranking
    from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
    candidates = [
        RankedCandidate(name=f, score=s, group=detector.detect_gender_proxy(request.resume_texts[f]))
        for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    fcr = FairnessConstrainedRanker(threshold=0.8)
    report = fcr.rerank(candidates)

    return {
        "bias_audit": audit,
        "fcr_report": {
            "original_air": report.original_air,
            "final_air": report.final_air,
            "num_swaps": report.num_swaps,
            "displacement_cost": report.displacement_cost,
            "fairness_satisfied": report.fairness_satisfied,
            "original_ranking": report.original_ranking,
            "fair_ranking": report.fair_ranking,
            "group_stats": report.group_stats,
            "pareto_points": report.pareto_points,
        },
    }


@app.post("/counterfactual")
async def counterfactual_explain(request: ExplainRequest):
    """
    Generate counterfactual explanation for a candidate.
    Shows which missing skills would most improve their ranking.
    """
    from explainability.counterfactual import CounterfactualExplainer

    explainer = CounterfactualExplainer()
    manager = get_embedding_manager()

    # Compute score
    jd_emb = manager.sbert_model.encode(request.jd_text, convert_to_numpy=True)
    resume_emb = manager.sbert_model.encode(request.resume_text, convert_to_numpy=True)
    score = float(manager.cosine_similarity(jd_emb, resume_emb))

    report = explainer.explain_candidate(
        candidate_name="uploaded_candidate",
        candidate_score=score,
        candidate_resume=request.resume_text,
        jd_text=request.jd_text,
        all_scores={"uploaded_candidate": score},
        top_k=5,
    )

    return {
        "candidate": report.candidate_name,
        "original_rank": report.original_rank,
        "original_score": report.original_score,
        "potential_best_rank": report.potential_best_rank,
        "skills_analyzed": report.total_skills_analyzed,
        "improvements": [
            {
                "skill": imp.skill,
                "score_delta": imp.score_delta,
                "rank_improvement": imp.rank_improvement,
                "counterfactual_score": imp.counterfactual_score,
            }
            for imp in report.top_improvements
        ],
        "summary": report.actionable_summary,
    }


# ─── Run Server ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FAIMR API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
