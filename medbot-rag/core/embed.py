from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_CONFIGS

_ENCODERS: Dict[str, SentenceTransformer] = {}
_ACTIVE_PROFILE: Optional[str] = None


def init_encoder_from_config(profile: str) -> SentenceTransformer:
    global _ENCODERS, _ACTIVE_PROFILE

    if profile not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown profile '{profile}'. Options: {list(EMBEDDING_CONFIGS.keys())}")

    if profile not in _ENCODERS:
        model_name = EMBEDDING_CONFIGS[profile]["model"]
        print(f"[embed] Loading encoder for profile '{profile}': {model_name}")
        _ENCODERS[profile] = SentenceTransformer(model_name)

    _ACTIVE_PROFILE = profile
    return _ENCODERS[profile]


def encode(text: str, profile: str) -> List[float]:
    encoder = init_encoder_from_config(profile)
    vec = encoder.encode([text])[0]
    return [float(x) for x in vec]


def encode_many(texts: List[str], profile: str) -> List[List[float]]:
    encoder = init_encoder_from_config(profile)
    vecs = encoder.encode(texts, batch_size=32, show_progress_bar=False)
    return [[float(x) for x in v] for v in vecs]


def embedding_dim(profile: str) -> int:
    if profile not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown profile '{profile}'")
    return EMBEDDING_CONFIGS[profile]["dim"]


def active_profile() -> Optional[str]:
    return _ACTIVE_PROFILE
