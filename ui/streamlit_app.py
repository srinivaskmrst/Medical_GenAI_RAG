"""Streamlit user interface for the medical RAG platform."""

from __future__ import annotations

import time
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="Medical GenAI RAG", page_icon="🩺", layout="wide")

DEFAULT_API_BASE = "http://localhost:8000/api/v1"

if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "health" not in st.session_state:
    st.session_state.health = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def api_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def call_health() -> dict:
    try:
        resp = requests.get(api_url("/health"), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def call_query(question: str, top_k: int) -> dict:
    resp = requests.post(
        api_url("/query"),
        json={
            "query": question,
            "top_k": top_k,
            "filters": {},
            "session_id": st.session_state.session_id,
        },
        timeout=300,
    )
    if resp.status_code == 400:
        raise ValueError(resp.json().get("detail", "Request rejected by a guardrail"))
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            score = src.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            st.write(
                f"- **{src.get('document_id', 'unknown')}** "
                f"(page {src.get('page_number', 'n/a')}, score {score_text})"
            )


with st.sidebar:
    st.title("🩺 Medical GenAI RAG")
    st.session_state.api_base = st.text_input("API base URL", value=st.session_state.api_base)
    if st.button("Check health", use_container_width=True):
        st.session_state.health = call_health()

    health = st.session_state.health
    if health:
        if health.get("error"):
            st.error(f"API unreachable: {health['error']}")
        else:
            def badge(ok: object) -> str:
                return "🟢" if ok else "🔴"

            st.write(f"{badge(health.get('qdrant'))} Qdrant (main)")
            st.write(f"{badge(health.get('cache'))} Qdrant (cache)")
            st.write(f"{badge(health.get('long_term_memory'))} Qdrant (long-term memory)")
            st.write(f"{badge(health.get('ollama'))} Ollama")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")
    top_k = st.slider("Retrieval top_k", min_value=1, max_value=20, value=5)
    if st.button("New session / clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

tab_ask, tab_ingest, tab_health = st.tabs(["💬 Ask", "📄 Ingest Documents", "🩺 Health"])

with tab_ask:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about the ingested medical documents…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")
            start = time.time()
            try:
                result = call_query(question, top_k)
            except ValueError as exc:
                placeholder.error(f"Rejected by guardrail: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Rejected by guardrail: {exc}"})
            except requests.RequestException as exc:
                placeholder.error(f"API error: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ API error: {exc}"})
            else:
                elapsed = time.time() - start
                metadata = result.get("retrieval_metadata", {})
                cache_hit = bool(metadata.get("cache_hit"))
                confidence = float(result.get("confidence", 0.0))
                answer = result.get("answer", "")

                placeholder.markdown(answer)
                status = "⚡ cache hit" if cache_hit else "🔎 generated"
                st.caption(f"{status} · confidence {confidence:.2f} · {elapsed:.1f}s")

                sources = result.get("sources", [])
                render_sources(sources)

                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with tab_ingest:
    st.subheader("Ingest documents into the vector store")
    st.caption("Points at the same synthetic corpus this project ships with by default.")
    path = st.text_input("File or directory path", value="data/Medical_Input_Data")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chunk (dry run)", use_container_width=True):
            try:
                resp = requests.post(api_url("/documents/ingest"), json={"file_path": path}, timeout=300)
                resp.raise_for_status()
                st.success(resp.json())
            except requests.RequestException as exc:
                st.error(str(exc))
    with col2:
        if st.button("Chunk + Index", type="primary", use_container_width=True):
            with st.spinner("Chunking and indexing… this can take a while for a full directory."):
                try:
                    resp = requests.post(api_url("/documents/index"), json={"file_path": path}, timeout=1800)
                    resp.raise_for_status()
                    st.success(resp.json())
                except requests.RequestException as exc:
                    st.error(str(exc))

    st.divider()
    st.subheader("Delete a document")
    doc_id = st.text_input("Document ID to delete")
    if st.button("Delete") and doc_id:
        try:
            resp = requests.delete(api_url(f"/documents/{doc_id}"), timeout=30)
            resp.raise_for_status()
            st.success(resp.json())
        except requests.RequestException as exc:
            st.error(str(exc))

with tab_health:
    st.subheader("System health")
    if st.button("Refresh", key="refresh_health"):
        st.session_state.health = call_health()
    elif st.session_state.health is None:
        st.session_state.health = call_health()

    health = st.session_state.health or {}
    if health.get("error"):
        st.error(f"API unreachable at {st.session_state.api_base}: {health['error']}")
    else:
        cols = st.columns(4)
        cols[0].metric("Qdrant (main)", "OK" if health.get("qdrant") else "DOWN")
        cols[1].metric("Qdrant (cache)", "OK" if health.get("cache") else "DOWN")
        cols[2].metric("Qdrant (long-term)", "OK" if health.get("long_term_memory") else "DOWN")
        cols[3].metric("Ollama", "OK" if health.get("ollama") else "DOWN")
        st.json(health)
