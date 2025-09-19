# src/config.py
# src/config.py
"""
Global configuration for MedBot RAG.
All parameters for ANN, threshold, filters, and index name
should be defined here instead of hard-coding elsewhere.
"""

# MongoDB index
INDEX_NAME = "chunks_vector_index"

# ANN parameters
NUM_CANDIDATES = 200    
TOP_K = 7              
THRESHOLD = 0.35      

# Default filter for vectorSearch
DEFAULT_FILTERS = {
    "task": "medical_dialogue",
    "source": "healthcaremagic"
}

# Logging settings
SAVE_VECTORS = True    
