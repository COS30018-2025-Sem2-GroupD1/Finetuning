import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from time import perf_counter
import math

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.db import get_collection
from core.embed import encode
from core.rag import vector_search, apply_threshold
from core.gemini import build_prompt, call_gemini_json
from config import INDEX_NAME, NUM_CANDIDATES, TOP_K  

#config
try:
    from config import THRESHOLD
except Exception:
    THRESHOLD = None

try:
    from config import DEFAULT_FILTERS
except Exception:
    DEFAULT_FILTERS = {}

try:
    from config import SAVE_VECTORS
except Exception:
    SAVE_VECTORS = False

app = FastAPI(title="MedBot RAG API")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _score_summary(hits: List[Dict[str, Any]]) -> Dict[str, float]:
    """Tóm tắt điểm _score để so sánh chi tiết."""
    s = [float(h.get("_score", 0.0) or 0.0) for h in hits]
    if not s:
        return {
            "max": 0.0, "mean": 0.0, "std": 0.0,
            "max_pct": 0.0, "mean_pct": 0.0,
        }
    n = len(s)
    mx = max(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n if n > 1 else 0.0
    std = math.sqrt(var)
    return {
        "max": round(mx, 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "max_pct": round(100.0 * mx, 2),
        "mean_pct": round(100.0 * mean, 2),
    }

def _filter_pct(col, filters: Dict[str, Any] | None) -> float:
    total = col.estimated_document_count() or 1
    eligible = col.count_documents(filters or {})
    return round(eligible * 100.0 / total, 2)

def _log_qna_min(
    db,
    *,
    question: str,
    q_vec,
    hits: List[Dict[str, Any]],
    answer_raw: str,
    answer_final: str,
    stats: Dict[str, Any],
    config: Dict[str, Any],
    save_vectors: bool = False,
    log_collection: str = "qa_collection",
):
    ctx = []
    for i, h in enumerate(hits):
        item = {
            "c_index": i,
            "_id": h.get("_id"),
            "parent_id": h.get("parent_id"),
            "chunk_index": h.get("chunk_index"),
            "source": h.get("source"),
            "score": float(h.get("_score", 0.0)),
            "text": (h.get("text") or "")[:1000],
        }
        if save_vectors and "vector" in h:
            item["vector"] = h["vector"]
        ctx.append(item)

    doc = {
        "created_at": _utcnow_iso(),
        "question": question,
        "q_index": (q_vec.tolist() if (q_vec is not None and save_vectors) else None),
        "answer": {"raw": answer_raw or "", "final": answer_final or ""},
        "contexts": ctx,
        "stats": stats,
        "config": config,
    }
    res = db[log_collection].insert_one(doc)
    return str(res.inserted_id)


# -------------------- Request/Response models --------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    locale: str = "en-AU"

    numCandidates: Optional[int] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None

    # logging & detail
    collection: str = "qa_collection"
    save_vectors: Optional[bool] = None
    detail: bool = True          
    ctx_chars: int = 1000     

class ChatResponse(BaseModel):
    config: Dict[str, Any]
    insufficient: bool
    stats: Dict[str, Any]
    answer: Dict[str, Any]
    filter_pct: float
    scores: Dict[str, Any]
    contexts: List[Dict[str, Any]]
    contexts_full: Optional[List[Dict[str, Any]]] = None
    log_id: Optional[str] = None


# --------------------------- Health ---------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# --------------------------- Main ----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    num_candidates = req.numCandidates or NUM_CANDIDATES
    top_k = req.top_k or TOP_K
    threshold = req.threshold if (req.threshold is not None) else THRESHOLD
    save_vectors = SAVE_VECTORS if (req.save_vectors is None) else req.save_vectors

    src_col = get_collection("medbot", "medical_chunks")
    db = src_col.database

    t0 = perf_counter()
    q_vec = encode(req.query)
    t1 = perf_counter()

    hits, latency_ms = vector_search(
        src_col,
        q_vec,
        top_k=top_k,
        num_candidates=num_candidates,
        filters=DEFAULT_FILTERS,
        index_name=INDEX_NAME,
        include_vector=save_vectors,
    )
    for i, h in enumerate(hits):
        h["c_index"] = i

    # 3) threshold + score summary
    insufficient, max_s, mean_s = apply_threshold(hits, threshold)
    score_sum = _score_summary(hits)
    filt_pct = _filter_pct(src_col, DEFAULT_FILTERS)

    # 4) LLM
    sys_text, user_text = build_prompt(req.query, hits, locale=req.locale)
    raw_text, parsed, usage, retries = call_gemini_json(sys_text, user_text)

    answer_final = (parsed.get("answer") if isinstance(parsed, dict) else None) or ""
    if not answer_final and isinstance(parsed, dict):
        answer_final = parsed.get("user_summary", "") or ""
    if not answer_final:
        answer_final = (raw_text or "").strip()

    stats = {
        "latency_ms_retrieval": latency_ms,
        "latency_ms_embed": round((t1 - t0) * 1000.0, 2),
        "retrieval_max_score": max_s,
        "retrieval_mean_score": mean_s,
        "retrieval_max_pct": (max_s or 0.0) * 100.0,
        "retrieval_mean_pct": (mean_s or 0.0) * 100.0,
        "llm_retries": retries,
        "llm_usage": usage or {},
    }
    cfg = {
        "top_k": top_k,
        "numCandidates": num_candidates,
        "threshold": threshold,
        "index": INDEX_NAME,
        "filters": DEFAULT_FILTERS,
        "detail": req.detail,
        "ctx_chars": req.ctx_chars,
        "save_vectors": save_vectors,
    }

    # 5) logging
    log_id = _log_qna_min(
        db=db,
        question=req.query,
        q_vec=q_vec,
        hits=hits,
        answer_raw=raw_text or "",
        answer_final=answer_final,
        stats={**stats, **score_sum},
        config=cfg,
        save_vectors=save_vectors,
        log_collection=req.collection,
    )

    # 6) Result
    preview_ctx: List[Dict[str, Any]] = [
        {
            "c_index": c.get("c_index"),
            "score": float(c.get("_score", 0.0)),
            "text": (c.get("text") or "")[:200],
        }
        for c in hits
    ]

    full_ctx: Optional[List[Dict[str, Any]]] = None
    if req.detail:
        lim = max(1, int(req.ctx_chars))
        full_ctx = []
        for c in hits:
            full_ctx.append({
                "c_index": c.get("c_index"),
                "_id": c.get("_id"),
                "parent_id": c.get("parent_id"),
                "chunk_index": c.get("chunk_index"),
                "source": c.get("source"),
                "score": float(c.get("_score", 0.0)),
                "text": (c.get("text") or "")[:lim],
            })

    return {
        "config": cfg,
        "insufficient": insufficient,
        "stats": stats,
        "answer": {"final": answer_final, "raw": raw_text},
        "filter_pct": filt_pct,
        "scores": score_sum,
        "contexts": preview_ctx,
        "contexts_full": full_ctx,
        "log_id": log_id,
    }


@app.post("/query", response_model=ChatResponse)
def query_alias(req: ChatRequest):
    return chat(req)
