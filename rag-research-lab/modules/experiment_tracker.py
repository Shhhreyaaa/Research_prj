import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Path to the experiments CSV
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
CSV_PATH = os.path.join(CSV_DIR, "experiments.csv")

# Columns list
COLUMNS = [
    "Timestamp", 
    "Question", 
    "Chunk Size", 
    "Retrieval Latency", 
    "Generation Latency", 
    "Total Latency", 
    "Number of Retrieved Chunks"
]

def init_experiments_csv():
    """
    Ensures that the experiments directory and experiments.csv file exist.
    Creates the file with header columns if it does not exist.
    """
    os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)

def log_experiment(
    question: str, 
    chunk_size: int, 
    retrieval_latency: float, 
    generation_latency: float, 
    num_retrieved_chunks: int
) -> Dict[str, Any]:
    """
    Logs an individual experiment execution to the CSV.
    
    Args:
        question: User query string.
        chunk_size: The chunk size setting used.
        retrieval_latency: Vector similarity search latency.
        generation_latency: Gemini response generation latency.
        num_retrieved_chunks: Number of chunks retrieved.
        
    Returns:
        The logged experiment entry dict.
    """
    init_experiments_csv()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_latency = retrieval_latency + generation_latency
    
    row = [
        timestamp,
        question,
        chunk_size,
        round(retrieval_latency, 4),
        round(generation_latency, 4),
        round(total_latency, 4),
        num_retrieved_chunks
    ]
    
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
    # Return as dict
    return dict(zip(COLUMNS, row))

def get_experiments() -> pd.DataFrame:
    """
    Reads the experiments CSV file and returns it as a pandas DataFrame.
    If the file doesn't exist, returns an empty DataFrame with the proper columns.
    """
    init_experiments_csv()
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

def clear_experiments() -> None:
    """
    Clears all recorded experiments by resetting the CSV file.
    """
    os.makedirs(CSV_DIR, exist_ok=True)
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
