"""Streamlit UI: upload → ingest → chat con citas."""
import streamlit as st
import requests
import tempfile
import subprocess
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hybrid RAG Engine", page_icon="🔍", layout="wide")
st.title("🔍 Hybrid RAG Engine")
st.caption("BM25 + Vector retrieval · CrossEncoder reranking · Verified citations")

with st.sidebar:
    st.header("📄 Load Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Ingest", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            for f in uploaded:
                (Path(tmp) / f.name).write_bytes(f.read())
            with st.spinner("Indexing..."):
                result = subprocess.run(
                    ["python", "-m", "src.ingestion.pipeline", "--docs", tmp],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success(f"✓ {result.stdout.strip()}")
            else:
                st.error(result.stderr)

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**, p. {s['page']} — score: `{s['score']:.3f}`")

if question := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            try:
                r = requests.post(f"{API_URL}/query", json={"question": question, "k": 5})
                data = r.json()
                st.markdown(data["answer"])
                with st.expander("📎 Sources"):
                    for s in data["sources"]:
                        st.markdown(f"- **{s['source']}**, p. {s['page']} — score: `{s['score']:.3f}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"]
                })
            except Exception as e:
                st.error(f"API error: {e}")
