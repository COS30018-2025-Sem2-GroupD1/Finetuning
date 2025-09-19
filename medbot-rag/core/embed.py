import os
import typing as t

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception: 
    torch = None 

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ENCODER: SentenceTransformer | None = None


def _pick_device() -> str:
    prefer = os.getenv("EMBED_DEVICE", "").lower()
    if prefer in {"cuda", "cpu", "mps"}:
        return prefer
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    if (
        torch is not None
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def get_encoder() -> SentenceTransformer:
    global _ENCODER
    if _ENCODER is None:
        name = os.getenv("EMBED_MODEL", _DEFAULT_MODEL)
        _ENCODER = SentenceTransformer(name, device=_pick_device())
    return _ENCODER


def embedding_dim() -> int:
    return get_encoder().get_sentence_embedding_dimension()


def encode(text: str) -> np.ndarray:
    vec = get_encoder().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32, copy=False)


def encode_many(texts: t.Sequence[str], batch_size: int = 64) -> np.ndarray:
    vecs = get_encoder().encode(
        list(texts),
        convert_to_numpy=True,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32, copy=False)


def get_model() -> SentenceTransformer:  
    return get_encoder()


def embed_query(text: str) -> list[float]:
    return encode(text).tolist()


def embed_corpus(texts: t.Sequence[str], batch_size: int = 64) -> list[list[float]]:
    return encode_many(texts, batch_size=batch_size).tolist()
