import time
import os
from typing import List, Tuple
import requests
from langchain_core.documents import Document

def generate_answer(
    query: str, 
    retrieved_docs: List[Document], 
    api_key: str = None
) -> Tuple[str, float]:
    """
    Generates an answer using Google Gemini REST API based on the retrieved document chunks.
    Measures and returns the model API call latency.
    
    Args:
        query: User question.
        retrieved_docs: List of retrieved LangChain Documents.
        api_key: Optional Gemini API key. If not provided, it looks in environment variables.
        
    Returns:
        A tuple of (generated_answer, generation_latency_seconds).
    """
    # 1. Resolve API key
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("Gemini API Key is not set. Please provide it in the sidebar or configure it in the .env file.")
        
    # 2. Format context from documents
    formatted_contexts = []
    for idx, doc in enumerate(retrieved_docs):
        source_info = f"Source {idx+1} | Document: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'Unknown')}"
        formatted_contexts.append(f"[{source_info}]\n{doc.page_content}\n")
    
    context_str = "\n".join(formatted_contexts)
    
    # 3. Build prompt
    prompt = f"""You are a professional AI research assistant. Answer the user's question about the research paper based only on the provided context excerpts. If the context does not contain enough information to answer the question, state that clearly (do not make up information). Be precise, scientific, and reference the Source number where applicable.

Context Excerpts:
---------------------
{context_str}
---------------------

Question: {query}

Answer:"""

    # 4. Invoke model and measure response latency
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={resolved_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            answer_text = f"API Error (HTTP {response.status_code}): {response.text}"
        else:
            response_json = response.json()
            # Extract response text from Gemini's JSON structure
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                    answer_text = candidate["content"]["parts"][0]["text"]
                else:
                    answer_text = f"API Error: Unexpected response format. Response: {response_json}"
            else:
                answer_text = f"API Error: No content candidates returned. Response: {response_json}"
            
    except Exception as e:
        # In case of network connection or timeout issues, return the error message
        answer_text = f"Network/Connection Error: {str(e)}"
        
    end_time = time.perf_counter()
    generation_latency = end_time - start_time
    
    # Return response text and time taken
    return answer_text, generation_latency
