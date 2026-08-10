"""Decoupled Presentation Layer — Main Entrypoint Dashboard.

Communicates purely via HTTP and WebSockets with the headless FastAPI backend.
Can be safely removed or swapped with any React/Next.js frontend.
"""

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Enterprise RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Enterprise LangChain & LangGraph RAG Studio")
st.markdown("""
Welcome to the **Enterprise RAG Starter Kit**. This presentation layer is completely decoupled from 
the core backend engine (`src/`) and interacts solely through standard HTTP REST and WebSocket APIs.
""")

st.divider()

# Backend Health & Readiness Check
st.subheader("System & Infrastructure Status")

col1, col2, col3 = st.columns(3)

try:
    with httpx.Client(timeout=3.0) as client:
        resp = client.get(f"{API_BASE_URL}/health")
        if resp.status_code == 200:
            health_data = resp.json()
            infra = health_data.get("infrastructure", {})

            with col1:
                pg_status = infra.get("postgres_pgvector", "unknown")
                if pg_status == "connected":
                    st.success("✔ PostgreSQL (pgvector): Connected")
                else:
                    st.error("✖ PostgreSQL (pgvector): Disconnected")

            with col2:
                redis_status = infra.get("redis_semantic_cache", "unknown")
                if redis_status == "connected":
                    st.success("✔ Redis Semantic Cache: Connected")
                else:
                    st.warning("⚠ Redis Semantic Cache: Disconnected")

            with col3:
                model = health_data.get("default_model", "N/A")
                st.info(f"🤖 Default Model: `{model}`")

        else:
            st.warning(f"Backend responded with status code: {resp.status_code}")

except Exception as err:
    st.error(
        f"Backend is currently unreachable at `{API_BASE_URL}`. "
        f"Make sure to start the API server with: `make run-api`"
    )

st.divider()

st.subheader("Available Workspaces")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    ### 📂 Data Ingestion Portal
    - Drag & drop PDF, DOCX, TXT, MD, and CSV files
    - **Visual Extraction Preview**: Inspect extracted Markdown tables before indexing
    - **Hierarchical (Small-to-Big) Chunking**: Index child chunks in `pgvector` and parent sections in PostgreSQL
    - Real-time indexing progress bar and deduplication checks
    """)
    st.page_link("pages/1_📂_Data_Loader.py", label="Open Data Loader", icon="📂")

with c2:
    st.markdown("""
    ### 💬 Conversational RAG Studio
    - Multi-turn conversation powered by **12-Step LangGraph Agent Pipeline**
    - Dynamic model switcher (OpenAI, Anthropic, Gemini, Groq, Ollama)
    - **Bidirectional WebSocket Streaming**: Real-time status frames, tokens, and kill switch
    - Interactive expandable citation drawer and **Waterfall Latency Breakdown**
    """)
    st.page_link("pages/2_💬_RAG_Chat.py", label="Open RAG Chat", icon="💬")
