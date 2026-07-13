from typing import Any, Dict, List

import streamlit as st


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False
    st.session_state.run_llm = None

if not st.session_state.backend_ready:
    st.info("Booting the knowledge base and Groq model. This may take a moment...")
    with st.spinner("Loading embeddings, vector index, and backend services..."):
        try:
            from core import run_llm as backend_run_llm
        except Exception as e:
            st.error("The backend could not be initialized.")
            st.exception(e)
            st.stop()

        st.session_state.run_llm = backend_run_llm
        st.session_state.backend_ready = True

# Cost warning
st.warning(
    "⚠️ **Cost Notice**: This app uses Groq API for responses. "
    "You must provide your own GROQ_API_KEY. "
    "Free tier: ~5,000 requests/month. "
    "Paid usage charged at ~$0.05–$0.20 per 1M tokens. "
    "[Get free API key](https://console.groq.com/keys)"
)

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = st.session_state.run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)