from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import time


def _to_list_floats(vec: Any) -> List[float]:
    try:
        if hasattr(vec, "tolist"):
            return [float(x) for x in vec.tolist()]
        return [float(x) for x in vec]
    except Exception:
        raise TypeError("query vector must be iterable (list/tuple/numpy array) of numbers")


def vector_search(
    col,
    query_vector: Any,
    *,
    top_k: int,
    num_candidates: int,
    filters: Optional[Dict[str, Any]] = None,
    index_name: str,
    include_vector: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    qvec = _to_list_floats(query_vector)
    flt = filters or {}

    project_fields = {
        "_id": 1,
        "parent_id": 1,
        "chunk_index": 1,
        "source": 1,
        "text": 1,
        "score": {"$meta": "vectorSearchScore"},
    }
    if include_vector:
        project_fields["vector"] = 1

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "vector",
                "queryVector": qvec,
                "numCandidates": int(num_candidates),
                "limit": int(top_k),
                "filter": flt,
            }
        },
        {"$project": project_fields},
    ]

    t0 = time.perf_counter()
    docs = list(col.aggregate(pipeline))
    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000.0, 2)

    for d in docs:
        try:
            d["_score"] = float(d.get("score", 0.0) or 0.0)
        except Exception:
            d["_score"] = 0.0

    return docs, latency_ms


def apply_threshold(hits: List[Dict[str, Any]], threshold: Optional[float]) -> Tuple[bool, float, float]:
    scores = [float(h.get("_score", 0.0) or 0.0) for h in hits] if hits else []
    if not scores:
        return (True, 0.0, 0.0) if threshold is not None else (True, 0.0, 0.0)

    max_s = max(scores)
    mean_s = sum(scores) / len(scores)

    if threshold is None:
        return (False, max_s, mean_s)

    insufficient = max_s < float(threshold)
    return insufficient, max_s, mean_s
