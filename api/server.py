"""
ASRIS — REST API Server
FastAPI-based API for resume-JD matching, ranking, and explanations.

Usage:
    python -m api.server
    # or: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
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

app = FastAPI(
    title="ASRIS — AI Resume Screening API",
    description="Match resumes to job descriptions using multi-signal AI ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    return HTMLResponse(content="<h1>ASRIS API is running. Frontend not found.</h1>")


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
    resume_embs = manager.encode_sbert(resume_texts, use_cache=False)

    scored = []
    for filename, emb in resume_embs.items():
        score = float(manager.cosine_similarity(jd_emb, emb))
        scored.append((filename, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = [
        {"filename": f, "score": round(s, 4), "rank": i + 1}
        for i, (f, s) in enumerate(scored[:top_k])
    ]

    return {"ranked_candidates": results, "total_candidates": len(scored)}


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(request: RankRequest):
    """
    Rank resumes against a job description.
    Accepts a JD and a dict of resumes, returns ranked candidates.
    """
    if not request.resume_texts:
        raise HTTPException(status_code=400, detail="No resume texts provided")

    manager = get_embedding_manager()

    # Encode JD
    jd_emb = manager.sbert_model.encode(request.jd_text, convert_to_numpy=True)

    # Encode resumes
    resume_embs = manager.encode_sbert(request.resume_texts, use_cache=False)

    # Score and rank
    scored = []
    for filename, emb in resume_embs.items():
        score = float(manager.cosine_similarity(jd_emb, emb))
        scored.append((filename, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:request.top_k]

    results = [
        RankResult(filename=f, score=round(s, 4), rank=i + 1)
        for i, (f, s) in enumerate(top_k)
    ]

    return RankResponse(
        ranked_candidates=results,
        total_candidates=len(scored),
    )


@app.post("/explain")
async def explain_match(request: ExplainRequest):
    """
    Explain why a resume matches (or doesn't match) a job description.
    Returns skill analysis, keyword overlap, and a human-readable verdict.
    """
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


# ─── Run Server ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting ASRIS API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
