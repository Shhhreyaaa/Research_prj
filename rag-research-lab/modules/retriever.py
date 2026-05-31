import time
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def retrieve_relevant_chunks(vector_store: FAISS, query: str, top_k: int = 4) -> Tuple[List[Document], float]:
    """
    Retrieves the most similar chunks for a given query and measures similarity search latency.
    
    Args:
        vector_store: The active FAISS vector store.
        query: The user question/query string.
        top_k: Number of chunks to retrieve.
        
    Returns:
        A tuple containing:
        - List of retrieved Document objects.
        - Similarity search/retrieval latency in seconds (float).
    """
    start_time = time.perf_counter()
    
    # Perform similarity search
    retrieved_docs = vector_store.similarity_search(query, k=top_k)
    
    end_time = time.perf_counter()
    retrieval_latency = end_time - start_time
    
    return retrieved_docs, retrieval_latency
