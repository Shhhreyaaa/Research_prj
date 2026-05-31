from langchain_huggingface import HuggingFaceEmbeddings
import time
from typing import Dict, Any, Tuple

# Constants for the specified model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings object.
    Uses the local CPU device for computing embeddings.
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def get_embedding_metadata() -> Dict[str, Any]:
    """
    Returns metadata about the embedding model.
    """
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "dimension": EMBEDDING_DIMENSION
    }
