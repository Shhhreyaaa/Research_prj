import fitz  # PyMuPDF
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any
import io

def load_pdf(pdf_file, file_name: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Load a PDF file and extract text page-by-page.
    
    Args:
        pdf_file: File-like object (e.g. Streamlit UploadedFile) or bytes containing PDF data.
        file_name: The name of the file (used for metadata and reporting).
        
    Returns:
        A tuple containing:
        - List of LangChain Document objects (one per page).
        - A dictionary with statistics: file_name, num_pages, total_words.
    """
    # If it's a file-like object (Streamlit's UploadedFile), read its bytes.
    # We copy or read to ensure we don't affect stream position if read elsewhere,
    # though usually calling read() is fine.
    if hasattr(pdf_file, "read"):
        # Make sure to read from start
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
    else:
        pdf_bytes = pdf_file

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    documents = []
    total_words = 0
    num_pages = len(doc)
    
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Calculate words on this page
        word_count = len(text.split())
        total_words += word_count
        
        # Create LangChain Document object
        metadata = {
            "source": file_name,
            "page": page_num + 1,
            "total_pages": num_pages
        }
        documents.append(Document(page_content=text, metadata=metadata))
        
    stats = {
        "file_name": file_name,
        "num_pages": num_pages,
        "total_words": total_words
    }
    
    return documents, stats
