# MongoDB namespace
DB_NAME = "rag_med"

EMBEDDING_CONFIGS = {
    # 1) GEMMA (medical)
    "gemma": {
        "collection": "chunks_gemma_med",
        "index": "vector_index_gemma", 
        "dim": 1024,                    
        "hf_model": "sentence-transformers/embeddinggemma-300m-medical",
    },

    # 2) MedEmbed (large)
    "medembed": {
        "collection": "chunks_medembed_lg",
        "index": "vector_index_medembed",
        "dim": 768,
        "hf_model": "abhinand/MedEmbed-large-v0.1",
    },

    # 3) S-PubMedBERT (MS MARCO)
    "spubmed": {
        "collection": "chunks_spubmed_ms",
        "index": "vector_index_spubmed",
        "dim": 768,
        "hf_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    },
}

DEFAULT_EMBED_PROFILE = "gemma"

NUM_CANDIDATES = 200   
TOP_K = 7              
THRESHOLD = 0.35      
DEFAULT_FILTERS = {}   

# Logging
SAVE_VECTORS = True    
