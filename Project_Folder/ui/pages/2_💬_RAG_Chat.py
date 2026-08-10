"""RAG Chat Studio — Bidirectional WebSocket Streaming, Citations & Telemetry."""

import asyncio
import json
import uuid
import httpx
import streamlit as st
import websockets

API_BASE_URL = "http://localhost:8000/api/v1"
WS_URL = "ws://localhost:8000/ws/chat"

st.set_page_config(page_title="RAG Chat Studio", page_icon="💬", layout="wide")

st.title("💬 Conversational RAG Studio")
st.caption("Powered by 12-Step LangGraph Agent Pipeline with Strict Remote Prompt Hub & Bidirectional WebSockets")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

# Sidebar Controls
st.sidebar.header("⚙️ Model & Generation Controls")

# Fetch available models from backend
available_providers = ["openai", "anthropic", "gemini", "groq", "ollama"]
selected_provider = st.sidebar.selectbox("LLM Provider", available_providers, index=0)

models_map = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-3-5-sonnet", "claude-3-haiku"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
    "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "ollama": ["mistral", "llama3", "qwen2.5"],
}
selected_model = st.sidebar.selectbox("Active Model", models_map.get(selected_provider, ["default"]))
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.divider()
st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
    st.rerun()


# Display Conversation History
for msg in st.session_state.messages:
    role = msg.get("role")
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))

        # Display citations if attached
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Sources & Citations ({len(citations)})"):
                for idx, c in enumerate(citations, 1):
                    st.markdown(
                        f"**[{idx}] {c.get('source_name')}** (Page {c.get('page_number')}, Score: `{c.get('relevance_score')}`)\n"
                        f"> {c.get('snippet')}"
                    )

        # Display telemetry if attached
        telemetry = msg.get("telemetry")
        if telemetry:
            with st.expander("📊 Execution Metrics & Waterfall Spans"):
                t_cols = st.columns(4)
                t_cols[0].metric("Total Latency", f"{telemetry.get('total_latency_ms', 0)} ms")
                t_cols[1].metric("TTFT", f"{telemetry.get('ttft_ms', 'N/A')} ms")
                tokens = telemetry.get("tokens", {})
                t_cols[2].metric("Total Tokens", tokens.get("total_tokens", 0))
                cost = telemetry.get("estimated_cost_usd", 0.0)
                t_cols[3].metric("Est. Cost", f"${cost:.5f}")

                spans = telemetry.get("waterfall_spans", [])
                if spans:
                    st.markdown("**Span Latency Waterfall:**")
                    for s in spans:
                        st.caption(f"• `{s.get('name')}`: **{s.get('duration_ms')} ms**")


# Asynchronous WebSocket Stream Runner
async def stream_rag_chat(prompt_text: str, response_placeholder, status_placeholder):
    full_response = ""
    citations_data = []
    telemetry_data = {}

    try:
        async with websockets.connect(WS_URL) as ws:
            # 1. Send query frame
            query_frame = {
                "action": "query",
                "user_query": prompt_text,
                "session_id": st.session_state.session_id,
                "model_provider": selected_provider,
                "model_name": selected_model,
                "temperature": temperature,
            }
            await ws.send(json.dumps(query_frame))

            # 2. Receive streaming events
            while True:
                msg_raw = await ws.recv()
                msg = json.loads(msg_raw)
                event = msg.get("event")

                if event == "status":
                    status_placeholder.info(f"⏳ {msg.get('data')}")

                elif event == "token":
                    status_placeholder.empty()
                    full_response += msg.get("data", "")
                    response_placeholder.markdown(full_response + "▌")

                elif event == "citations":
                    citations_data = msg.get("data", [])

                elif event == "telemetry":
                    telemetry_data = msg.get("data", {})

                elif event == "cancelled":
                    status_placeholder.warning("🛑 Generation halted via kill switch.")
                    break

                elif event == "error":
                    status_placeholder.error(f"Error: {msg.get('message')}")
                    break

                elif event == "done":
                    response_placeholder.markdown(full_response)
                    status_placeholder.empty()
                    break

    except Exception as err:
        status_placeholder.error(f"WebSocket error: {err}")

    return full_response, citations_data, telemetry_data


# Chat Input
user_input = st.chat_input("Ask a question about your indexed documents...")

if user_input:
    # 1. Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Render assistant response via WebSocket streaming
    with st.chat_message("assistant"):
        status_box = st.empty()
        response_box = st.empty()

        # Run async stream
        response_text, citations_res, telemetry_res = asyncio.run(
            stream_rag_chat(user_input, response_box, status_box)
        )

        # Save assistant message to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "citations": citations_res,
            "telemetry": telemetry_res,
        })
        st.rerun()
