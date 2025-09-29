from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
from time import perf_counter
import math

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.db import get_collection
from core.embed import encode
from core.rag import vector_search, apply_threshold
from core.gemini import build_prompt, call_gemini_json

from config import (
    DB_NAME,
    EMBEDDING_CONFIGS,      
    DEFAULT_EMBED_PROFILE,    
    NUM_CANDIDATES,
    TOP_K,
    THRESHOLD,
    DEFAULT_FILTERS,
    SAVE_VECTORS,
)

app = FastAPI(title="MedBot RAG API")


# ----------------------------- Helpers -----------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_summary(hits: List[Dict[str, Any]]) -> Dict[str, float]:
    s = [float(h.get("_score", 0.0) or 0.0) for h in hits]
    if not s:
        return {"max": 0.0, "mean": 0.0, "std": 0.0, "max_pct": 0.0, "mean_pct": 0.0}
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
) -> str:
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


def _resolve_profile(p: Optional[str]) -> str:
    prof = (p or DEFAULT_EMBED_PROFILE).strip()
    if prof not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown profile '{prof}'. Options: {list(EMBEDDING_CONFIGS.keys())}")
    return prof


def _store_for_profile(profile: str) -> Tuple[str, str, str]:
    cfg = EMBEDDING_CONFIGS[profile]
    return DB_NAME, cfg["collection"], cfg["index"]


# ----------------------------- Schemas -----------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    locale: str = "en-AU"

    profile: Optional[str] = None

    numCandidates: Optional[int] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None

    collection: str = "qa_collection"
    save_vectors: Optional[bool] = None


class ChatResponse(BaseModel):
    config: Dict[str, Any]
    insufficient: bool
    stats: Dict[str, Any]

    answer: Dict[str, Any]

    filter_pct: float
    scores: Dict[str, Any]

    contexts: List[Dict[str, Any]]

    log_id: Optional[str] = None


class MultiChatResponse(BaseModel):
    results: List[ChatResponse]


# ----------------------------- Core Endpoint -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) Resolve runtime parameters
    num_candidates = req.numCandidates or NUM_CANDIDATES
    top_k = req.top_k or TOP_K
    threshold = req.threshold if (req.threshold is not None) else THRESHOLD
    save_vectors = SAVE_VECTORS if (req.save_vectors is None) else req.save_vectors

    profile = _resolve_profile(req.profile)
    db_name, col_name, index_name = _store_for_profile(profile)

    # 2) Prepare collection & embed the query with the selected profile
    src_col = get_collection(db_name, col_name)
    db = src_col.database

    t0 = perf_counter()
    q_vec = encode(req.query, profile=profile)
    t1 = perf_counter()

    hits, retrieval_latency_ms = vector_search(
        src_col,
        q_vec,
        top_k=top_k,
        num_candidates=num_candidates,
        filters=DEFAULT_FILTERS,
        index_name=index_name,
        include_vector=save_vectors,
    )
    for i, h in enumerate(hits):
        h["c_index"] = i

    # 3) Threshold & score summaries
    if threshold is None:
        insufficient, max_s, mean_s = (False, 0.0, 0.0)
    else:
        insufficient, max_s, mean_s = apply_threshold(hits, threshold)

    scores = _score_summary(hits)
    filt_pct = _filter_pct(src_col, DEFAULT_FILTERS)

    # 4) Build prompt + call LLM
    sys_text, user_text = build_prompt(req.query, hits, locale=req.locale)
    raw_text, parsed, usage, retries = call_gemini_json(sys_text, user_text)

    # 5) Select final answer with robust fallbacks
    answer_final = (parsed.get("answer") if isinstance(parsed, dict) else None) or ""
    if not answer_final and isinstance(parsed, dict):
        answer_final = parsed.get("user_summary", "") or ""
    if not answer_final:
        answer_final = (raw_text or "").strip()

    # 6) Stats + config snapshot
    stats = {
        "latency_ms_retrieval": retrieval_latency_ms,
        "latency_ms_embed": round((t1 - t0) * 1000.0, 2),
        "retrieval_max_score": max_s,
        "retrieval_mean_score": mean_s,
        "retrieval_max_pct": (max_s or 0.0) * 100.0,
        "retrieval_mean_pct": (mean_s or 0.0) * 100.0,
        "llm_retries": retries,
        "llm_usage": usage or {},
    }
    cfg = {
        "profile": profile,
        "db": db_name,
        "collection": col_name,
        "index": index_name,
        "top_k": top_k,
        "numCandidates": num_candidates,
        "threshold": threshold,
        "filters": DEFAULT_FILTERS,
        "save_vectors": save_vectors,
    }

    # 7) Logging to Mongo
    log_id = _log_qna_min(
        db=db,
        question=req.query,
        q_vec=q_vec,
        hits=hits,
        answer_raw=raw_text or "",
        answer_final=answer_final,
        stats={**stats, **scores},
        config=cfg,
        save_vectors=save_vectors,
        log_collection=req.collection,
    )

    # 8) Build API response (contexts = full preview up to 1000 chars)
    contexts: List[Dict[str, Any]] = []
    for c in hits:
        contexts.append({
            "c_index": c.get("c_index"),
            "_id": c.get("_id"),
            "parent_id": c.get("parent_id"),
            "chunk_index": c.get("chunk_index"),
            "source": c.get("source"),
            "score": float(c.get("_score", 0.0)),
            "text": (c.get("text") or "")[:1000],
        })

    return {
        "config": cfg,
        "insufficient": insufficient,
        "stats": stats,
        "answer": {"final": answer_final, "raw": raw_text},
        "filter_pct": filt_pct,
        "scores": scores,
        "contexts": contexts,
        "log_id": log_id,
    }


# ----------------------------- Convenience -----------------------------

@app.post("/query", response_model=ChatResponse)
def query_alias(req: ChatRequest):
    """Alias of /chat to keep backwards compatibility."""
    return chat(req)


@app.post("/chat_multi", response_model=MultiChatResponse)
def chat_multi(req: ChatRequest):
    """
    Run the same query across multiple profiles (default: all).
    Useful for side-by-side comparisons of retrieval/LLM behavior.
    """
    profiles = list(EMBEDDING_CONFIGS.keys()) if not req.profile else [req.profile]
    results: List[ChatResponse] = []
    for p in profiles:
        sub_req = ChatRequest(
            query=req.query,
            locale=req.locale,
            profile=p,
            numCandidates=req.numCandidates,
            top_k=req.top_k,
            threshold=req.threshold,
            collection=req.collection,
            save_vectors=req.save_vectors,
        )
        results.append(chat(sub_req))  
    return {"results": results}
