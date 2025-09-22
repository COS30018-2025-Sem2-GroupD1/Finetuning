from __future__ import annotations
from typing import Iterable, List, Optional
import numpy as np

try:
    import torch
except Exception:
    torch = None

from sentence_transformers import SentenceTransformer

_ENCODER: Optional[SentenceTransformer] = None
_MODEL_NAME: Optional[str] = None


def _pick_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"


# Encoder lifecycle
def set_encoder(model_name: str) -> None:
    global _ENCODER, _MODEL_NAME
    _MODEL_NAME = model_name
    _ENCODER = SentenceTransformer(model_name, device=_pick_device())


def get_encoder() -> SentenceTransformer:
    if _ENCODER is None:
        raise RuntimeError("Encoder not initialized. Call set_encoder(model_name) first.")
    return _ENCODER


# Normalization helpers
def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if X.ndim == 1:
        denom = np.linalg.norm(X) + eps
        return (X / denom).astype(np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return (X / denom).astype(np.float32)


# Encoding APIs
def encode_many(
    texts: Iterable[str],
    batch_size: int = 128,
    normalize: bool = True,
    to_numpy: bool = True,
) -> np.ndarray:
    """
    Encode a batch of texts. Returns np.ndarray [N, D], float32.
    - normalize=True ensures cosine-friendly vectors.
    """
    enc = get_encoder()
    emb = enc.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    emb = emb.astype(np.float32, copy=False)
    if normalize:
        emb = _l2_normalize(emb)
    return emb if to_numpy else emb 


def encode_one(
    text: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a single text to shape [D], float32.
    """
    vec = encode_many([text], batch_size=1, normalize=normalize)
    return vec[0]


def embedding_dim() -> int:
    """
    Inspect the current encoder to get output embedding dimension.
    """
    enc = get_encoder()
    # Robust way: encode a tiny sample and read shape
    sample = enc.encode(["_"], convert_to_numpy=True, normalize_embeddings=False)
    if isinstance(sample, np.ndarray):
        return int(sample.shape[-1])
    sample = np.asarray(sample)
    return int(sample.shape[-1])


def current_model_name() -> Optional[str]:
    """
    Return the last model_name passed to set_encoder, for logging.
    """
    return _MODEL_NAME
