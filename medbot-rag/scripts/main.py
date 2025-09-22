# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import time

from db import get_collection
from embed import set_encoder, encode_one
from rag import vector_search_for_model, to_contexts
from gemini import build_prompt, call_gemini
from config import EMBEDDING_CONFIGS, NUM_CANDIDATES, TOP_K, DEFAULT_FILTERS

app = FastAPI(title="RAG Chat API", version="2.0")


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    model_key: str            
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    num_candidates: Optional[int] = None


class ChatResponse(BaseModel):
    query: str
    model_key: str
    answer: str
    contexts: List[Dict[str, Any]]
    latency_ms: int


# API endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model_key not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown model_key: {req.model_key}")

    cfg = EMBEDDING_CONFIGS[req.model_key]
    top_k = req.top_k or TOP_K
    num_cand = req.num_candidates or NUM_CANDIDATES
    filters = req.filters or DEFAULT_FILTERS

    set_encoder(cfg["hf_model"])

    qvec = encode_one(req.query)

    col = get_collection("rag_med", cfg["collection"])
    hits, lat = vector_search_for_model(
        db_name="rag_med",
        model_key=req.model_key,
        query_vec=qvec,
        top_k=top_k,
        num_candidates=num_cand,
        filters=filters,
    )

    contexts = to_contexts(hits, max_chars=700, keep_vector=False)

    prompt = build_prompt(req.query, contexts)
    answer = call_gemini(prompt)

    return ChatResponse(
        query=req.query,
        model_key=req.model_key,
        answer=answer,
        contexts=contexts,
        latency_ms=int(lat),
    )


# Entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
