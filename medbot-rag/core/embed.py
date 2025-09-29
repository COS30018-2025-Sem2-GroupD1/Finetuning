from __future__ import annotations
from typing import Dict, Optional, List, Iterable

# Torch to detect device
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_CONFIGS

# Cache for all loaded encoders
_ENCODERS: Dict[str, SentenceTransformer] = {}
_CURRENT_KEY: Optional[str] = None


# Device selection

def _device() -> str:
    if _HAS_TORCH:
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def _resolve_model_name(profile: str) -> str:
    cfg = EMBEDDING_CONFIGS.get(profile, {})
    name = cfg.get("model") or cfg.get("hf_model")
    if not name:
        raise KeyError(f"Profile '{profile}' is missing 'hf_model' or 'model'.")
    return name


# Encoder lifecycle
def init_encoder_from_config(profile: str) -> SentenceTransformer:
    if profile not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown profile '{profile}'. Options: {list(EMBEDDING_CONFIGS.keys())}")
    if profile not in _ENCODERS:
        model_name = _resolve_model_name(profile)
        print(f"[embed] Loading encoder for profile '{profile}': {model_name} on {_device()}")
        _ENCODERS[profile] = SentenceTransformer(model_name, device=_device())
    return _ENCODERS[profile]


def set_encoder(key_or_model: str) -> SentenceTransformer:
    global _CURRENT_KEY
    if key_or_model in EMBEDDING_CONFIGS:
        enc = init_encoder_from_config(key_or_model)
        _CURRENT_KEY = key_or_model
        return enc

    custom_key = f"__custom__::{key_or_model}"
    if custom_key not in _ENCODERS:
        print(f"[embed] Loading custom encoder: {key_or_model} on {_device()}")
        _ENCODERS[custom_key] = SentenceTransformer(key_or_model, device=_device())
    _CURRENT_KEY = custom_key
    return _ENCODERS[custom_key]


def get_encoder(key: Optional[str] = None) -> SentenceTransformer:
    use_key = key or _CURRENT_KEY
    if not use_key:
        raise RuntimeError("No encoder selected. Call set_encoder(profile) first.")
    if use_key in _ENCODERS:
        return _ENCODERS[use_key]
    if use_key in EMBEDDING_CONFIGS:
        return init_encoder_from_config(use_key)
    raise RuntimeError(f"Encoder '{use_key}' not initialized.")


def clear_encoders() -> None:
    _ENCODERS.clear()
    global _CURRENT_KEY
    _CURRENT_KEY = None


# Encoder information
def embedding_dim(key: Optional[str] = None) -> int:
    use_key = key or _CURRENT_KEY
    if not use_key:
        raise RuntimeError("Call set_encoder(...) first.")
    enc = get_encoder(use_key)
    try:
        return int(enc.get_sentence_embedding_dimension())
    except Exception:
        pass
    if use_key in EMBEDDING_CONFIGS and "dim" in EMBEDDING_CONFIGS[use_key]:
        return int(EMBEDDING_CONFIGS[use_key]["dim"])
    return 768


def encoder_name(key: Optional[str] = None) -> str:
    enc = get_encoder(key)
    return getattr(enc, "model_name_or_path", "unknown")


# Encoding functions
def encode_one(text: str, key: Optional[str] = None, normalize: bool = True) -> List[float]:
    enc = get_encoder(key)
    vec = enc.encode(text, normalize_embeddings=normalize)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def encode_batch(
    texts: List[str],
    key: Optional[str] = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> List[List[float]]:
    enc = get_encoder(key)
    vecs = enc.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
    return vecs.tolist() if hasattr(vecs, "tolist") else [list(v) for v in vecs]


# Compatibility wrappers (for main.py and run_ingest.py)
def encode(text: str, *, profile: Optional[str] = None, normalize: bool = True) -> List[float]:
    if profile:
        set_encoder(profile)
    return encode_one(text, key=None, normalize=normalize)


def encode_many(
    texts: Iterable[str],
    *,
    profile: Optional[str] = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> List[List[float]]:
    if profile:
        set_encoder(profile)
    return encode_batch(list(texts), key=None, batch_size=batch_size, normalize=normalize)


# Public API
__all__ = [
    "set_encoder", "get_encoder", "init_encoder_from_config", "clear_encoders",
    "embedding_dim", "encoder_name",
    "encode_one", "encode_batch",
    "encode", "encode_many",
]
