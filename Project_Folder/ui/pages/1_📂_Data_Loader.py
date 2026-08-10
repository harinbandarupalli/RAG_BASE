"""Data Loader Portal — Visual Extraction, Markdown Table Preview & Indexing."""

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Data Loader", page_icon="📂", layout="wide")

st.title("📂 Document Ingestion & Chunking Portal")
st.markdown("""
Upload documents to run layout extraction, convert tables into Markdown format,
apply **Hierarchical (Small-to-Big)** chunking, and dual-index into **pgvector** and **BM25**.
""")

st.divider()

# 1. File Upload Section
uploaded_file = st.file_uploader(
    "Choose a file to ingest",
    type=["pdf", "txt", "md", "csv", "docx"],
    help="Supported formats: PDF (with tables & OCR), Markdown, Plain Text, CSV, Word DOCX",
)

force_reindex = st.checkbox("Force Reindex (Bypass SHA-256 duplicate detection)", value=False)

if uploaded_file is not None:
    st.info(f"📄 Selected file: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    col1, col2 = st.columns([1, 1])

    # 2. Preview Action
    with col1:
        if st.button("🔍 Inspect Layout & Preview Tables", use_container_width=True):
            with st.spinner("Extracting layout and markdown tables..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(f"{API_BASE_URL}/ingest/preview", files=files)

                    if resp.status_code == 200:
                        preview_data = resp.json()
                        st.success(
                            f"✔ Extracted {preview_data.get('total_pages', 0)} pages, "
                            f"{preview_data.get('tables_detected', 0)} tables detected."
                        )

                        st.subheader("Extracted Section Preview")
                        for sec in preview_data.get("sections", []):
                            with st.expander(f"Page {sec['page_number']} — {sec['char_count']} chars (Tables: {sec['has_tables']})"):
                                st.markdown(sec["snippet"])
                    else:
                        st.error(f"Preview failed: {resp.text}")

                except Exception as err:
                    st.error(f"Error connecting to backend: {err}")

    # 3. Ingestion Action
    with col2:
        if st.button("🚀 Index Document to pgvector", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Commencing pipeline...")
            with st.spinner("Extracting, chunking, and indexing into PostgreSQL..."):
                try:
                    progress_bar.progress(25, text="Running visual table extraction...")
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"force_reindex": force_reindex}

                    progress_bar.progress(50, text="Hierarchical chunking and pgvector embedding...")
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(f"{API_BASE_URL}/ingest", files=files, params=params)

                    progress_bar.progress(100, text="Indexing complete!")

                    if resp.status_code == 200:
                        res = resp.json()
                        status = res.get("status")

                        if status == "skipped":
                            st.warning(f"⚠ {res.get('message')} (SHA-256: `{res.get('file_hash')[:12]}...`)")
                        elif status == "success":
                            st.balloons()
                            st.success(f"✔ Document **{res.get('file_name')}** indexed successfully in {res.get('elapsed_ms')}ms!")

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Pages Extracted", res.get("pages_extracted", 0))
                            m2.metric("Tables Found", res.get("tables_detected", 0))
                            m3.metric("Child Chunks in pgvector", res.get("child_chunks_indexed", 0))
                            m4.metric("Parent Docs in Postgres", res.get("parent_documents_stored", 0))

                            st.caption(f"Document ID: `{res.get('doc_id')}` | SHA-256: `{res.get('file_hash')}`")
                        else:
                            st.info(res.get("message", "Processing completed."))
                    else:
                        st.error(f"Ingestion failed: {resp.text}")

                except Exception as err:
                    st.error(f"Error communicating with ingestion API: {err}")
