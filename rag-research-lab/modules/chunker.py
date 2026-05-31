import time
from typing import List, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Split standard LangChain Document objects into smaller chunks using recursive text splitting.
    
    Args:
        documents: A list of Document objects.
        chunk_size: Target size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        A tuple containing:
        - List of chunked Document objects.
        - Dictionary of statistics (chunk_size, chunk_overlap, total_chunks, chunking_time_seconds).
    """
    start_time = time.perf_counter()
    
    # We use characters for RecursiveCharacterTextSplitter.
    # This splitter splits by characters, which is standard. 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    end_time = time.perf_counter()
    chunking_time = end_time - start_time
    
    stats = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "chunking_time_seconds": chunking_time
    }
    
    return chunks, stats
