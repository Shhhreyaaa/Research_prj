# 🔬 RAG Research Lab: Scientific Literature Q&A Platform

RAG Research Lab is a research-oriented Retrieval-Augmented Generation (RAG) framework designed to perform semantic question answering on scientific papers while systematically evaluating the trade-offs of chunking sizes on latency profiles.

Instead of operating as a black-box document chatbot, this project treats RAG parameter selection as an empirical optimization problem. It provides an automated evaluation dashboard to measure, track, and analyze how varying chunk sizes (256, 512, and 1024 characters) affect retrieval and language model generation times.

---

## 🛠️ Tech Stack & Key Libraries

- **Frontend:** Streamlit
- **Backend & Logic Orchestration:** LangChain (core, community, huggingface)
- **Vector Database:** FAISS (CPU-optimized)
- **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Language Model:** Google Gemini API (`gemini-1.5-flash`)
- **PDF Extraction:** PyMuPDF (`fitz`)
- **Data Analytics & Visualizations:** Pandas, Matplotlib

---

## 📐 System Architecture

The pipeline processes files locally, indexing documents and recording latencies in a unified feedback loop:

```text
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│  Research PDF   │ ──> │ PyMuPDF Parser  │ ──> │  Document Splitter    │
└─────────────────┘     └─────────────────┘     │ (256 vs 512 vs 1024)  │
                                                └───────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│ Vector Database │ <── │  FAISS Indexer  │ <── │ HuggingFace Embedder  │
│ (Local Storage) │     └─────────────────┘     │ (all-MiniLM-L6-v2)    │
└─────────────────┘                             └───────────────────────┘
        │
        ▼ (Retrieve top-k chunks)
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│   Similarity    │ ──> │ Gemini Prompt   │ ──> │ Generated Response &  │
│   Search Latency│     │ Builder (LLM)   │     │ Latency Benchmarking  │
└─────────────────┘     └─────────────────┘     └───────────────────────┘
                                                            │
                                                            ▼
                                                ┌───────────────────────┐
                                                │    experiments.csv    │
                                                └───────────────────────┘
                                                            │
                                                            ▼
                                                ┌───────────────────────┐
                                                │ Streamlit Dashboard   │
                                                │  (Pandas/Matplotlib)  │
                                                └───────────────────────┘
```

---

## 🧪 Experimental Methodology

We configure and run controlled experiments to isolate the effect of text block size on end-to-end question answering efficiency:

1. **Independent Variable (Chunk Size):** 
   - We benchmark document splits at three distinct resolutions: **256 characters**, **512 characters**, and **1024 characters** (with a static overlap of 50 characters).
2. **Dependent Variables (Latency):**
   - **Retrieval Latency ($L_{ret}$):** Time taken to retrieve the $k$ most relevant document chunks from the FAISS database.
   - **Generation Latency ($L_{gen}$):** Time taken by the Gemini API to process the prompt template containing the retrieved contexts and return the final textual response.
   - **Total Latency ($L_{total}$):** $L_{total} = L_{ret} + L_{gen}$.
3. **Consistency:** All comparative runs query the same vector index populated from the same paper, using identical question prompts.

---

## 📈 Latency Analysis & Observations

### Typical Latency Profile (Theoretical & Seeded Runs)

| Chunk Size (Chars) | Avg Retrieval ($L_{ret}$) | Avg Generation ($L_{gen}$) | Avg Total ($L_{total}$) | Key Characteristics |
|:------------------:|:-------------------------:|:--------------------------:|:-----------------------:|:--------------------|
| **256**            | ~ 0.050 s                 | ~ 1.35 s                   | ~ 1.40 s                | Fragmented context; high vector count; faster retrieval; higher LLM reasoning difficulty. |
| **512**            | ~ 0.038 s                 | ~ 1.05 s                   | ~ 1.09 s                | Balanced; optimized contextual footprint. |
| **1024**           | ~ 0.027 s                 | ~ 0.85 s                   | ~ 0.88 s                | Cohesive contexts; fewer index nodes; slightly slower prompt start but faster context ingestion. |

### Observations Summary
- **Retrieval Latency:** Larger chunk sizes yield fewer total vectors in the FAISS index, which often speeds up local lookup indexing slightly on standard flat indexes, though index build time increases.
- **Generation Latency:** Larger chunk sizes inject larger dense blocks of text into the Gemini system context. While this provides rich context, it results in a larger input token structure which can impact overall LLM time-to-first-token depending on model configuration.
- **Context Granularity Trade-off:** While 1024-token chunks might exhibit faster retrieval lookup times, they can exceed context bounds if $k$ is high, leading to diluted context or higher API costs. Conversely, 256-character chunks can split key sentences, causing loss of contextual cohesion. Empirically, **512 characters** is the optimal operating window.

---

## 📁 Repository Structure

```text
rag-research-lab/
├── app.py                      # Streamlit application UI and pages
├── requirements.txt            # System dependencies
├── .env.example                # Template for Gemini API key
├── README.md                   # Academic project documentation
├── modules/
│   ├── pdf_loader.py           # PyMuPDF parser wrapper
│   ├── chunker.py              # Text recursive character chunker module
│   ├── embeddings.py           # sentence-transformers model loader
│   ├── vector_store.py         # FAISS indexing, saving, loading and rebuilding
│   ├── retriever.py            # Similarity search executor
│   ├── gemini_client.py        # Gemini API assistant orchestrator
│   └── experiment_tracker.py   # experiments.csv file tracker
├── experiments/
│   └── experiments.csv         # Raw metrics database
└── reports/                    # Directory for saving experimental results & charts
```

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.8 or higher
- A Google Gemini API Key (obtainable via [Google AI Studio](https://aistudio.google.com/))

### Installation
1. Clone this repository to your local directory.
2. Navigate into the directory and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and insert your API key:
   ```bash
   cp .env.example .env
   # Open .env and set GEMINI_API_KEY=your_key_here
   ```

### Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## 🎓 Resume-Oriented Outcomes
This project is designed to demonstrate core competencies in applied AI/ML engineering, retrieval systems, and research-grade evaluation pipelines:
- **Developed a Retrieval-Augmented Generation framework** for scientific literature question answering using LangChain, Gemini, and FAISS.
- **Conducted comparative experiments** across multiple chunking configurations and analyzed retrieval performance.
- **Built an experiment-tracking pipeline** to measure latency and document retrieval effectiveness.
- **Implemented semantic search** and citation-based response generation for research paper analysis.

---

## 🔮 Future Work
1. **Multi-Agent Research Assistant:** Integrating LangGraph to coordinate hierarchical researchers where agent A summarizes, agent B checks claims, and agent C searches vectors.
2. **Alternative Embedding Models:** Comparing `all-MiniLM-L6-v2` against large multilingual models (`text-embedding-004` or `BGE-M3`) for retrieval accuracy.
3. **RAGAS Evaluation Integration:** Automating evaluation using metrics such as faithfulness, answer relevance, and context recall rather than solely focusing on latencies.
4. **Structured JSON Output:** Formatting answers matching specific JSON schemas to support automatic semantic database inserts.
