import time
import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> Tuple[FAISS, Dict[str, Any]]:
    """
    Build a FAISS vector store from document chunks and measure creation latency.
    
    Args:
        chunks: List of chunked Document objects.
        embeddings: HuggingFaceEmbeddings object.
        
    Returns:
        A tuple containing:
        - The initialized FAISS vector store.
        - Dictionary of statistics (num_vectors, index_creation_time_seconds).
    """
    start_time = time.perf_counter()
    
    # Build vector store using standard FAISS class method
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    end_time = time.perf_counter()
    creation_time = end_time - start_time
    
    stats = {
        "num_vectors": len(chunks),
        "index_creation_time_seconds": creation_time
    }
    
    return vector_store, stats

def save_vector_store(vector_store: FAISS, folder_path: str, index_name: str = "faiss_index") -> None:
    """
    Save the FAISS vector store index to disk.
    """
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path, index_name=index_name)

def load_vector_store(folder_path: str, embeddings: HuggingFaceEmbeddings, index_name: str = "faiss_index") -> FAISS:
    """
    Load the FAISS vector store index from disk.
    Note: allow_dangerous_deserialization=True is required to load locally saved FAISS indices in newer LangChain community.
    """
    return FAISS.load_local(
        folder_path, 
        embeddings, 
        index_name=index_name, 
        allow_dangerous_deserialization=True
    )
