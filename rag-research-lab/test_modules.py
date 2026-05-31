import os
import sys

# Add current directory to path to ensure modules directory is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.pdf_loader import load_pdf
    from modules.chunker import chunk_documents
    from modules.embeddings import get_embeddings_model, get_embedding_metadata
    from modules.vector_store import build_vector_store
    from modules.retriever import retrieve_relevant_chunks
    from modules.gemini_client import generate_answer
    from modules.experiment_tracker import log_experiment, get_experiments, clear_experiments
    print("SUCCESS: All modular imports completed successfully!")
except Exception as e:
    print(f"ERROR: Import failed. Details: {e}")
    sys.exit(1)
