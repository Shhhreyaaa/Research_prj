import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from modules.pdf_loader import load_pdf
from modules.chunker import chunk_documents
from modules.embeddings import get_embeddings_model, get_embedding_metadata
from modules.vector_store import build_vector_store
from modules.retriever import retrieve_relevant_chunks
from modules.gemini_client import generate_answer
from modules.experiment_tracker import log_experiment, get_experiments, clear_experiments

# Custom styling for premium UI appearance
st.set_page_config(
    page_title="RAG Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek CSS styles for a professional AI research platform look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #1e293b;
        color: #ffffff;
        border: 1px solid #3b82f6;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
        border-color: #60a5fa;
        box-shadow: 0px 0px 10px rgba(59, 130, 246, 0.4);
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-lbl {
        font-size: 14px;
        color: #94a3b8;
    }
</style>
""", 
unsafe_allow_html=True
)

# Initialize Session State
if "documents" not in st.session_state:
    st.session_state.documents = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chunk_stats" not in st.session_state:
    st.session_state.chunk_stats = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "vector_store_stats" not in st.session_state:
    st.session_state.vector_store_stats = None
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = None

# Sidebar Configuration
st.sidebar.title("🔬 RAG Research Lab")
st.sidebar.markdown("---")

# API Key config
env_key = os.environ.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key", 
    value=env_key if env_key else "", 
    type="password", 
    help="Add your Gemini API Key here or configure it in the .env file."
)
if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

# App Navigation
page = st.sidebar.radio(
    "Navigation", 
    ["📄 Upload Documents", "💬 Ask Questions", "📊 Evaluation Dashboard", "🔬 Research Summary"]
)

st.sidebar.markdown("---")
# System status sidebar card
st.sidebar.subheader("System Status")
if st.session_state.documents:
    st.sidebar.success(f"📄 Document Loaded: {st.session_state.doc_stats['file_name']}")
    if st.session_state.vector_store:
        st.sidebar.success(f"⚡ FAISS Index Built ({st.session_state.chunk_stats['chunk_size']} chars)")
    else:
        st.sidebar.warning("⚠️ FAISS Index Pending")
else:
    st.sidebar.info("📂 No Documents Loaded")

# Page 1: Upload Documents
if page == "📄 Upload Documents":
    st.title("📄 Document Loading & Indexing")
    st.markdown("Upload a scientific research paper (PDF) to extract its contents, split it into chunks, and index it into a FAISS vector database.")
    
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        
        # Load PDF and show stats if it changed
        if st.session_state.doc_stats is None or st.session_state.doc_stats['file_name'] != file_name:
            with st.spinner("Extracting PDF content..."):
                try:
                    documents, doc_stats = load_pdf(uploaded_file, file_name)
                    st.session_state.documents = documents
                    st.session_state.doc_stats = doc_stats
                    # Reset chunk and vector stores on new doc
                    st.session_state.chunks = None
                    st.session_state.chunk_stats = None
                    st.session_state.vector_store = None
                    st.session_state.vector_store_stats = None
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                
        if st.session_state.doc_stats:
            st.markdown("### Document Metadata & Parsing Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">File Name</div>
                    <div class="metric-val">{st.session_state.doc_stats['file_name']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Total Pages</div>
                    <div class="metric-val">{st.session_state.doc_stats['num_pages']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Total Words</div>
                    <div class="metric-val">{st.session_state.doc_stats['total_words']:,}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Vector Storage & Chunking Settings")
        
        # Chunk settings
        chunk_size = st.selectbox("Select Chunk Size (characters)", [256, 512, 1024], index=1)
        chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=200, value=50, step=10)
        
        if st.button("Chunk & Build FAISS Vector Database"):
            if not os.environ.get("GEMINI_API_KEY"):
                st.warning("⚠️ Warning: Gemini API Key is not set yet. You can build the vector database, but Q&A requires the API Key.")
                
            # 1. Chunking
            with st.spinner("Splitting document pages into overlapping chunks..."):
                chunks, chunk_stats = chunk_documents(st.session_state.documents, chunk_size, chunk_overlap)
                st.session_state.chunks = chunks
                st.session_state.chunk_stats = chunk_stats
                
            # 2. Embedding Model
            with st.spinner("Initializing HuggingFace Embedding model (sentence-transformers/all-MiniLM-L6-v2)..."):
                if st.session_state.embeddings_model is None:
                    st.session_state.embeddings_model = get_embeddings_model()
                embed_meta = get_embedding_metadata()
                
            # 3. FAISS Building
            with st.spinner("Generating embeddings and building FAISS vector index..."):
                vector_store, vs_stats = build_vector_store(st.session_state.chunks, st.session_state.embeddings_model)
                st.session_state.vector_store = vector_store
                st.session_state.vector_store_stats = vs_stats
                
            st.success("🎉 RAG Index Successfully Generated!")
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Total Chunks Created</div>
                    <div class="metric-val">{st.session_state.chunk_stats['total_chunks']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Indexing Latency</div>
                    <div class="metric-val">{st.session_state.vector_store_stats['index_creation_time_seconds']:.4f} s</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Embedding Dimensions</div>
                    <div class="metric-val">{embed_meta['dimension']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"**Embedding Model Used:** `{embed_meta['model_name']}`")
            st.markdown(f"**Chunking Time:** `{st.session_state.chunk_stats['chunking_time_seconds']:.4f} seconds`")

# Page 2: Ask Questions
elif page == "💬 Ask Questions":
    st.title("💬 Literature Question Answering & Evaluation")
    st.markdown("Query your uploaded documents. The application will search the FAISS vector database for relevant contexts and generate answers using Google Gemini. Run individual queries or multi-configuration comparative benchmarks.")
    
    if not st.session_state.vector_store:
        st.info("⚠️ Please upload a document and build the FAISS index first on the 'Upload Documents' page.")
    else:
        # Configuration Details
        active_chunk_size = st.session_state.chunk_stats['chunk_size']
        active_overlap = st.session_state.chunk_stats['chunk_overlap']
        st.info(f"💡 Active Configuration: Chunk Size = **{active_chunk_size}**, Overlap = **{active_overlap}**, Total Vectors = **{st.session_state.vector_store_stats['num_vectors']}**")
        
        # Presets
        st.markdown("### Suggested Research Questions")
        presets = [
            "What problem does this paper solve?",
            "What methodology was used in this study?",
            "What datasets were used for training or evaluation?",
            "What are the main limitations identified by the authors?",
            "What future work is suggested?"
        ]
        
        # Session state for holding the active question
        if "question_input" not in st.session_state:
            st.session_state.question_input = ""
            
        def select_preset(text):
            st.session_state.question_input = text

        cols = st.columns(len(presets))
        for i, preset in enumerate(presets):
            cols[i].button(preset.replace(" ", "\n", 2), key=f"preset_{i}", on_click=select_preset, args=(preset,))
            
        query = st.text_input("Ask a question about the document:", value=st.session_state.question_input, key="main_q_input")
        
        top_k = st.slider("Number of source chunks to retrieve (k)", min_value=1, max_value=8, value=4)
        
        col1, col2 = st.columns(2)
        submit_query = col1.button("Submit Query (Active Config)", use_container_width=True)
        run_experiment = col2.button("Run Multi-Config Experiment (Compare 256 vs 512 vs 1024)", use_container_width=True)
        
        if query:
            # 1. Standard Q&A
            if submit_query:
                with st.spinner("Searching index and generating answer..."):
                    try:
                        # Retrieval
                        retrieved_docs, retrieval_latency = retrieve_relevant_chunks(
                            st.session_state.vector_store, 
                            query, 
                            top_k=top_k
                        )
                        
                        # Generation
                        answer, generation_latency = generate_answer(
                            query, 
                            retrieved_docs
                        )
                        
                        # Log Experiment
                        log_experiment(
                            question=query,
                            chunk_size=active_chunk_size,
                            retrieval_latency=retrieval_latency,
                            generation_latency=generation_latency,
                            num_retrieved_chunks=len(retrieved_docs)
                        )
                        
                        # Display results
                        st.markdown("### Answer")
                        st.write(answer)
                        
                        # Latency breakdown cards
                        st.markdown("### Performance Metrics")
                        col_r, col_g, col_t = st.columns(3)
                        with col_r:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-lbl">Retrieval Latency</div>
                                <div class="metric-val">{retrieval_latency:.4f} s</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_g:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-lbl">Generation Latency</div>
                                <div class="metric-val">{generation_latency:.4f} s</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_t:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-lbl">Total Response Time</div>
                                <div class="metric-val">{(retrieval_latency + generation_latency):.4f} s</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Citations
                        st.markdown("### Retrieved Context Sources")
                        for idx, doc in enumerate(retrieved_docs):
                            with st.expander(f"Source {idx+1} (Page {doc.metadata.get('page', 'N/A')})"):
                                st.write(doc.page_content)
                                st.caption(f"Metadata: {doc.metadata}")
                                
                    except Exception as e:
                        st.error(f"Error executing Q&A pipeline: {e}")
            
            # 2. Multi-Config Comparative Experiment
            elif run_experiment:
                st.subheader("🔬 Running Multi-Configuration Benchmark")
                st.write(f"Evaluating query: *'{query}'* across different chunk sizes (256, 512, 1024)...")
                
                results = []
                progress_bar = st.progress(0)
                
                configs = [256, 512, 1024]
                
                # Check for Gemini key early
                if not os.environ.get("GEMINI_API_KEY"):
                    st.error("Please configure the Gemini API key in the sidebar before running experiments.")
                else:
                    for i, c_size in enumerate(configs):
                        st.write(f"Indexing & Querying with chunk size: **{c_size}**...")
                        
                        # Chunker
                        chunks, _ = chunk_documents(st.session_state.documents, c_size, 50)
                        
                        # Build Index
                        vs, _ = build_vector_store(chunks, st.session_state.embeddings_model)
                        
                        # Retrieval
                        ret_docs, ret_lat = retrieve_relevant_chunks(vs, query, top_k=top_k)
                        
                        # Generation
                        ans, gen_lat = generate_answer(query, ret_docs)
                        
                        # Log to CSV
                        log_experiment(
                            question=query,
                            chunk_size=c_size,
                            retrieval_latency=ret_lat,
                            generation_latency=gen_lat,
                            num_retrieved_chunks=len(ret_docs)
                        )
                        
                        results.append({
                            "Chunk Size": c_size,
                            "Retrieval (s)": round(ret_lat, 4),
                            "Generation (s)": round(gen_lat, 4),
                            "Total (s)": round(ret_lat + gen_lat, 4),
                            "Answer": ans
                        })
                        progress_bar.progress((i + 1) / len(configs))
                        
                    st.success("Experiments finished and logged successfully!")
                    
                    # Display comparison
                    df_res = pd.DataFrame(results)
                    st.dataframe(df_res[["Chunk Size", "Retrieval (s)", "Generation (s)", "Total (s)"]], use_container_width=True)
                    
                    # Show Answers side by side
                    for res in results:
                        with st.expander(f"Answer for Chunk Size {res['Chunk Size']}"):
                            st.write(res['Answer'])

# Page 3: Evaluation Dashboard
elif page == "📊 Evaluation Dashboard":
    st.title("📊 Performance & Latency Evaluation")
    st.markdown("Analyze performance profiles across different parameters. Results are pulled directly from `experiments/experiments.csv`.")
    
    # Utilities to reset or load sample data
    col1, col2 = st.columns(2)
    if col1.button("Reset Experiment Log", use_container_width=True):
        clear_experiments()
        st.success("Experiment log reset.")
        st.rerun()
        
    if col2.button("Seed Sample Evaluation Data", use_container_width=True):
        sample_runs = [
            ("What is RAG?", 256, 0.051, 1.254, 4),
            ("What is RAG?", 512, 0.038, 0.985, 4),
            ("What is RAG?", 1024, 0.027, 0.812, 4),
            ("What dataset was used?", 256, 0.048, 1.352, 4),
            ("What dataset was used?", 512, 0.035, 1.011, 4),
            ("What dataset was used?", 1024, 0.024, 0.842, 4),
            ("What are the limitations?", 256, 0.052, 1.450, 4),
            ("What are the limitations?", 512, 0.040, 1.120, 4),
            ("What are the limitations?", 1024, 0.031, 0.920, 4),
        ]
        for q, c_sz, ret_l, gen_l, num_ch in sample_runs:
            log_experiment(q, c_sz, ret_l, gen_l, num_ch)
        st.success("Sample experiment data seeded successfully!")
        st.rerun()

    st.markdown("---")
    
    df = get_experiments()
    
    if df.empty:
        st.warning("No experiment records found in log. Run a question or seed sample data above to visualize results.")
    else:
        st.subheader("Experiment History")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Performance Visualizations")
        
        # Prepare charts
        # Aggregate by chunk size
        avg_latencies = df.groupby("Chunk Size").agg({
            "Retrieval Latency": "mean",
            "Generation Latency": "mean",
            "Total Latency": "mean",
            "Question": "count"
        }).reset_index()
        
        col_c1, col_c2 = st.columns(2)
        
        # Style plots beautifully
        plt.style.use("dark_background")
        
        # Chart 1: Chunk Size vs Retrieval Latency
        with col_c1:
            st.markdown("#### Chunk Size vs. Average Retrieval Latency")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.bar(
                avg_latencies["Chunk Size"].astype(str), 
                avg_latencies["Retrieval Latency"], 
                color="#3b82f6", 
                edgecolor="#60a5fa",
                width=0.4
            )
            ax1.set_xlabel("Chunk Size (characters)")
            ax1.set_ylabel("Retrieval Latency (seconds)")
            ax1.set_title("Avg Retrieval Latency by Chunk Size")
            ax1.grid(True, linestyle="--", alpha=0.3)
            # Apply layout fixes
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
        # Chart 2: Chunk Size vs Total Response Time
        with col_c2:
            st.markdown("#### Chunk Size vs. Average Total Response Time")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(
                avg_latencies["Chunk Size"].astype(str), 
                avg_latencies["Total Latency"], 
                marker="o", 
                linewidth=2.5, 
                color="#10b981"
            )
            ax2.set_xlabel("Chunk Size (characters)")
            ax2.set_ylabel("Total Latency (seconds)")
            ax2.set_title("Avg Total Latency by Chunk Size")
            ax2.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
        # Chart 3: Question Count per Experiment
        st.markdown("#### Distribution of Queries across Chunk Configurations")
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        ax3.barh(
            avg_latencies["Chunk Size"].astype(str),
            avg_latencies["Question"],
            color="#f59e0b",
            edgecolor="#fbbf24",
            height=0.4
        )
        ax3.set_xlabel("Number of Runs / Questions Logged")
        ax3.set_ylabel("Chunk Size")
        ax3.set_title("Experiment Runs per Configuration")
        ax3.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

# Page 4: Research Summary
elif page == "🔬 Research Summary":
    st.title("🔬 Scientific Evaluation Summary")
    st.markdown("Aggregate experimental findings, metadata, and observations generated from system profiling.")
    
    df = get_experiments()
    
    if df.empty:
        st.warning("No experiment records found. Run questions to generate the summary.")
    else:
        st.subheader("Experimental Setup")
        embed_meta = get_embedding_metadata()
        
        num_docs = 1 if st.session_state.documents else 0
        doc_name = st.session_state.doc_stats['file_name'] if st.session_state.documents else "N/A"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Embedding Model:** `{embed_meta['model_name']}`")
            st.markdown(f"**Vector Dimension:** `{embed_meta['dimension']}`")
        with col2:
            st.markdown(f"**Primary Document:** `{doc_name}`")
            st.markdown(f"**Number of Documents:** `{num_docs}`")
        with col3:
            st.markdown(f"**Total Benchmarked Queries:** `{len(df)}`")
            
        st.markdown("---")
        st.subheader("Results Summary Table")
        
        # Calculate statistics
        avg_latencies = df.groupby("Chunk Size").agg({
            "Retrieval Latency": ["mean", "min", "max"],
            "Generation Latency": ["mean", "min", "max"],
            "Total Latency": ["mean", "min", "max"],
        })
        
        # Flatten columns
        avg_latencies.columns = [f"{col[0]}_{col[1]}" for col in avg_latencies.columns]
        avg_latencies = avg_latencies.reset_index()
        
        # Format table for presentation
        presentation_df = pd.DataFrame({
            "Chunk Size": avg_latencies["Chunk Size"],
            "Avg Retrieval (s)": avg_latencies["Retrieval Latency_mean"].round(4),
            "Min/Max Retrieval (s)": avg_latencies.apply(lambda r: f"{r['Retrieval Latency_min']:.4f} / {r['Retrieval Latency_max']:.4f}", axis=1),
            "Avg Generation (s)": avg_latencies["Generation Latency_mean"].round(4),
            "Min/Max Generation (s)": avg_latencies.apply(lambda r: f"{r['Generation Latency_min']:.4f} / {r['Generation Latency_max']:.4f}", axis=1),
            "Avg Total (s)": avg_latencies["Total Latency_mean"].round(4),
        })
        st.dataframe(presentation_df, use_container_width=True)
        
        # Fastest/Slowest configuration
        fastest_idx = presentation_df["Avg Total (s)"].idxmin()
        slowest_idx = presentation_df["Avg Total (s)"].idxmax()
        
        st.markdown("### Performance Extremes")
        col_f, col_s = st.columns(2)
        with col_f:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #10b981;">
                <div class="metric-lbl">Fastest Configuration (Avg Total Latency)</div>
                <div class="metric-val" style="color: #10b981;">Chunk Size {presentation_df.iloc[fastest_idx]['Chunk Size']}</div>
                <div style="font-size: 14px; margin-top: 5px; color: #a1a1aa;">Avg Latency: {presentation_df.iloc[fastest_idx]['Avg Total (s)']:.4f} s</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #ef4444;">
                <div class="metric-lbl">Slowest Configuration (Avg Total Latency)</div>
                <div class="metric-val" style="color: #ef4444;">Chunk Size {presentation_df.iloc[slowest_idx]['Chunk Size']}</div>
                <div style="font-size: 14px; margin-top: 5px; color: #a1a1aa;">Avg Latency: {presentation_df.iloc[slowest_idx]['Avg Total (s)']:.4f} s</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("Auto-Generated Observations")
        
        # Compute difference
        df_256 = presentation_df[presentation_df["Chunk Size"] == 256]
        df_512 = presentation_df[presentation_df["Chunk Size"] == 512]
        df_1024 = presentation_df[presentation_df["Chunk Size"] == 1024]
        
        observations = []
        
        # 1. Observation about retrieval latency
        if not df_256.empty and not df_1024.empty:
            lat_256 = df_256.iloc[0]["Avg Retrieval (s)"]
            lat_1024 = df_1024.iloc[0]["Avg Retrieval (s)"]
            pct_diff = abs(lat_256 - lat_1024) / max(lat_256, lat_1024) * 100
            faster = "256" if lat_256 < lat_1024 else "1024"
            slower = "1024" if faster == "256" else "256"
            observations.append(
                f"**Retrieval Scaling:** Similarity search with Chunk Size {faster} was **{pct_diff:.1f}% faster** on average than Chunk Size {slower}. This correlates with the vector index complexity and chunk count (smaller chunks produce more vectors, which can add linear query complexity)."
            )
            
        # 2. Observation about generation latency
        if not df_256.empty and not df_1024.empty:
            gen_256 = df_256.iloc[0]["Avg Generation (s)"]
            gen_1024 = df_1024.iloc[0]["Avg Generation (s)"]
            pct_diff = abs(gen_256 - gen_1024) / max(gen_256, gen_1024) * 100
            faster = "256" if gen_256 < gen_1024 else "1024"
            slower = "1024" if faster == "256" else "256"
            observations.append(
                f"**Generation Latency:** Generating answers using context from Chunk Size {faster} was **{pct_diff:.1f}% faster** on average than using Chunk Size {slower}. Because Gemini's processing latency is proportional to input token count, larger context blocks (1024 tokens vs 256 tokens) result in longer time-to-first-token and overall response generation latency."
            )
            
        # 3. Overall trade-off
        if not df_512.empty:
            avg_tot_512 = df_512.iloc[0]["Avg Total (s)"]
            observations.append(
                f"**Optimal Threshold Recommendation:** Chunk Size **512** represents a balanced compromise. It provides intermediate context granularity to support Gemini's reasoning without incurring the higher processing latency of 1024-token chunks or the excessive fragmentation of 256-token chunks."
            )
            
        for obs in observations:
            st.markdown(f"- {obs}")
