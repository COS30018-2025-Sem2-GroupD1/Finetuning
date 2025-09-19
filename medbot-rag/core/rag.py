from __future__ import annotations
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

from config import INDEX_NAME


def _as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x


def vector_search(
    col,
    query_vec,
    top_k: int,
    num_candidates: int,
    filters: Dict[str, Any] | None,
    index_name: str = INDEX_NAME,
    include_vector: bool = True,
) -> Tuple[List[Dict[str, Any]], float]:
    project = {
        "_id": 1,
        "text": 1,
        "_score": {"$meta": "searchScore"},
    }
    if include_vector:
        project["vector"] = 1

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "vector",
                "queryVector": _as_list(query_vec),
                "numCandidates": int(num_candidates),
                "limit": int(top_k),
                "filter": (filters or {}),
            }
        },
        {"$project": project},
    ]

    t0 = perf_counter()
    hits = list(col.aggregate(pipeline))
    latency_ms = (perf_counter() - t0) * 1000.0
    return hits, latency_ms


def apply_threshold(hits: Sequence[Dict[str, Any]], threshold: float) -> tuple[bool, float, float]:
    scores = [float(h.get("_score", 0.0)) for h in hits]
    if not scores:
        return True, 0.0, 0.0
    mx = max(scores)
    mean = sum(scores) / len(scores)
    return (mx < threshold), mx, mean


def to_contexts(
    hits: Sequence[Dict[str, Any]],
    max_chars: int = 700,
    keep_vector: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, 1):
        item = {
            "id": str(h.get("_id", f"ctx-{i}")),
            "text": (h.get("text") or "")[:max_chars],
            "_score": float(h.get("_score", 0.0)),
        }
        if keep_vector and "vector" in h:
            item["vector"] = h["vector"]
        out.append(item)
    return out
