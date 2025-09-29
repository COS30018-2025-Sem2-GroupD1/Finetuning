# run_ingest.py
from __future__ import annotations

import csv
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import ReplaceOne

# Project core utilities
from core.db import get_collection
from core.embed import encode_many

# Project configuration
from config import (
    DB_NAME,
    EMBEDDING_CONFIGS,      # e.g. {"gemma": {"collection": "...", "index": "...", "dim": ...}, ...}
    DEFAULT_FILTERS,        # e.g. {"task": "...", "source": "..."} (optional)
)

app = FastAPI(title="MedBot Ingest API")


# ----------------------------- Helpers -----------------------------

def _utcnow_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _extract_id(row: Dict[str, Any]) -> str:
    """
    Resolve a stable ID for a CSV record.
    Priority: id | _id | doc_id | uid; otherwise derive from content.
    """
    r = {(k or "").strip().lower(): v for k, v in row.items()}
    for key in ("id", "_id", "doc_id", "uid"):
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    base = (r.get("title") or r.get("question") or r.get("text") or "").strip()
    return f"auto-{abs(hash(base))}"


def _extract_text(row: Dict[str, Any]) -> str:
    """
    Extract the primary text for a record with graceful fallbacks.

    Fallback order:
      1) direct text fields: text | content | prompt | body
      2) instruction-tuning style: instruction + input + (ref) output[:400]
      3) QA style: question + (ref) answer[:200]
      4) title/summary/description
      5) empty string if nothing usable
    """
    r = {(k or "").strip().lower(): (v if v is not None else "") for k, v in row.items()}

    # 1) direct text fields
    for key in ("text", "content", "prompt", "body"):
        v = r.get(key, "")
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())

    # 2) instruction-tuning style
    instr = (r.get("instruction") or "").strip()
    inp   = (r.get("input") or "").strip()
    out   = (r.get("output") or "").strip()
    if instr or inp or out:
        parts = []
        if instr: parts.append(instr)
        if inp:   parts.append(inp)
        if out:   parts.append(f"(ref) {out[:400]}")
        return " ".join(" ".join(parts).split()).strip()

    # 3) QA style
    q = (r.get("question") or "").strip()
    a = (r.get("answer") or "").strip()
    if q or a:
        return " ".join((q, f"(ref) {a[:200]}" if a else "")).strip()

    # 4) metadata fields
    pieces = []
    for k in ("title", "summary", "description"):
        v = (r.get(k) or "").strip()
        if v:
            pieces.append(v)
    if pieces:
        return " ".join(" ".join(pieces).split())

    # 5) nothing usable
    return ""


def _normalize_multiline(s: str | None) -> str:
    """Trim lines and collapse redundant whitespace; remove empty lines."""
    if not s:
        return ""
    lines = [(" ".join(ln.split())).strip() for ln in s.splitlines()]
    return "\n".join([ln for ln in lines if ln])


def chunk_text(text: str, max_chars: int = 400, overlap: int = 50) -> List[str]:
    """
    Slice text into overlapping windows to preserve local context across chunk boundaries.

    Args:
        text: input string
        max_chars: window length
        overlap: characters shared between consecutive windows

    Returns:
        List of chunk strings.
    """
    s = _normalize_multiline(text)
    if not s:
        return []
    if max_chars <= 0:
        return [s]
    chunks, i = [], 0
    step = max(1, max_chars - max(0, overlap))
    while i < len(s):
        chunks.append(s[i:i + max_chars])
        i += step
    return chunks


# ----------------------- Request / Response Models -----------------------

class IngestCSVRequest(BaseModel):
    # Data source
    path: str = Field(..., description="Local path to CSV file")

    # Run one or many profiles in a single pass (default: all declared)
    profiles: List[str] = list(EMBEDDING_CONFIGS.keys())

    # Destination (collection is resolved per-profile)
    target_db: str = DB_NAME
    target_col: Optional[str] = None  # ignored when multi-profile

    # Chunking & batching knobs
    max_chars: int = 400
    overlap: int = 50
    batch_size: int = 64
    flush_n: int = 1000
    limit: Optional[int] = None

    # Augmentation handling (skip duplicated -aug variants by default)
    keep_aug: bool = False

    # Default metadata attached to each chunk
    tags: Optional[List[str]] = None
    task: str = Field(default=DEFAULT_FILTERS.get("task", "medical_dialogue"))
    source: str = Field(default=DEFAULT_FILTERS.get("source", "healthcaremagic"))


class IngestResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    profiles: Dict[str, Any]


# ----------------------------- Core Logic -----------------------------

def _build_items_from_csv(req: IngestCSVRequest) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load CSV once, normalize and chunk into item dicts (without vectors).

    Returns:
        items: list of documents ready for vectorization
        dropped_aug: count of skipped augmentation records
    """
    items: List[Dict[str, Any]] = []
    seen_root: set[str] = set()
    dropped_aug = 0

    with open(req.path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            if req.limit is not None and i >= req.limit:
                break

            rid = _extract_id(row)
            root = rid.split("-aug")[0]

            # Skip augmented duplicates unless explicitly kept
            if not req.keep_aug:
                if root in seen_root and ("-aug" in rid):
                    dropped_aug += 1
                    continue
                seen_root.add(root)

            text = _extract_text(row)
            if not text:
                continue

            parts = chunk_text(text, max_chars=req.max_chars, overlap=req.overlap)
            if not parts:
                continue

            for j, part in enumerate(parts):
                items.append({
                    "_id": f"{rid}#{j}",
                    "parent_id": rid,
                    "chunk_index": j,
                    "text": part,
                    "source": req.source,
                    "task": req.task,
                    "meta": {"tags": list(set(req.tags or []))},
                    "created_at": _utcnow_iso(),
                })

    return items, dropped_aug


def _embed_and_upsert(
    items: List[Dict[str, Any]],
    db_name: str,
    col_name: str,
    profile: str,
    batch_size: int,
    flush_n: int,
) -> int:
    """
    Vectorize by profile and upsert into the target collection in batches.

    Returns:
        Number of documents written.
    """
    col = get_collection(db_name, col_name)
    writes: List[ReplaceOne] = []
    written = 0
    batch: List[Dict[str, Any]] = []

    def flush_writes():
        nonlocal writes, written
        if not writes:
            return
        col.bulk_write(writes, ordered=False)
        written += len(writes)
        writes = []

    for it in items:
        batch.append(it)
        if len(batch) >= batch_size:
            texts = [x["text"] for x in batch]
            vecs = encode_many(texts, profile=profile)
            for x, v in zip(batch, vecs):
                doc = {**x, "vector": [float(z) for z in v]}
                writes.append(ReplaceOne({"_id": x["_id"]}, doc, upsert=True))
            batch = []
            if len(writes) >= flush_n:
                flush_writes()

    # Flush remainder
    if batch:
        texts = [x["text"] for x in batch]
        vecs = encode_many(texts, profile=profile)
        for x, v in zip(batch, vecs):
            doc = {**x, "vector": [float(z) for z in v]}
            writes.append(ReplaceOne({"_id": x["_id"]}, doc, upsert=True))
        batch = []
    flush_writes()

    return written


# ------------------------------ Endpoints ------------------------------

@app.post(
    "/ingest_csv",
    response_model=IngestResponse,
    summary="Ingest a CSV into multiple embedding-specific collections (single pass over data)",
)
def ingest_csv(req: IngestCSVRequest):
    # A) Build items once
    items, dropped_aug = _build_items_from_csv(req)
    if not items:
        return IngestResponse(
            ok=True,
            message="No rows/chunks produced from CSV.",
            profiles={p: {"chunks_written": 0} for p in req.profiles},
        )

    # B) Iterate profiles: encode + upsert
    results: Dict[str, Any] = {}
    for profile in req.profiles:
        cfg = EMBEDDING_CONFIGS.get(profile)
        if not cfg:
            results[profile] = {"error": f"Unknown profile '{profile}'. Options: {list(EMBEDDING_CONFIGS.keys())}"}
            continue

        col_name = cfg["collection"]
        written = _embed_and_upsert(
            items=items,
            db_name=req.target_db,
            col_name=col_name,
            profile=profile,
            batch_size=req.batch_size,
            flush_n=req.flush_n,
        )

        results[profile] = {
            "db": req.target_db,
            "collection": col_name,
            "chunks_written": written,
            "max_chars": req.max_chars,
            "overlap": req.overlap,
            "batch_size": req.batch_size,
            "flush_n": req.flush_n,
            "tags": req.tags or [],
            "task": req.task,
            "source": req.source,
            "dropped_aug": dropped_aug,
        }

    return IngestResponse(ok=True, profiles=results)


@app.get("/health")
def health():
    """Lightweight health-check endpoint."""
    return {"ok": True, "service": "ingest", "time": _utcnow_iso(), "profiles": list(EMBEDDING_CONFIGS.keys())}


# ---------------------------- Local Runner ----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_ingest:app", host="0.0.0.0", port=8001, reload=False)
