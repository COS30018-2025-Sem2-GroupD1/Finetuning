# Collection + Index mapping for each model
EMBEDDING_CONFIGS = {
    "gemma": {
        "collection": "chunks_gemma_med",
        "index": "vector_index_gemma",
        "dim": 1024,
        "hf_model": "sentence-transformers/embeddinggemma-300m-medical"
    },
    "medembed": {
        "collection": "chunks_medembed_lg",
        "index": "vector_index_medembed",
        "dim": 768,
        "hf_model": "abhinand/MedEmbed-large-v0.1"
    },
    "spubmed": {
        "collection": "chunks_spubmed_ms",
        "index": "vector_index_spubmed",
        "dim": 768,
        "hf_model": "pritamdeka/S-PubMedBert-MS-MARCO"
    }
}


# ANN parameters
NUM_CANDIDATES = 200
TOP_K = 7
THRESHOLD = 0.35

DEFAULT_FILTERS = {}
SAVE_VECTORS = True
