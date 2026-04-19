import streamlit as st
import sys, os
import warnings
import time
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from rag.chain import build_chain_with_sources
from rag.memory import ChatHistory


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="SEBI RAG Assistant", layout="wide")

st.markdown("""
# ⚖️ SEBI RAG Assistant  
<div style='color:gray;'>AI-powered assistant for SEBI regulations & compliance</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    llm_provider = st.selectbox("LLM Provider", ["ollama", "groq"])

    alpha = st.slider(
        "Hybrid Search Balance",
        0.0, 1.0, 0.4, 0.1,
        help="0 = keyword (BM25), 1 = semantic (vector)"
    )

    use_rerank = st.checkbox(
        "Enable Reranking (BGE)",
        value=True,
        help="Improves answer quality"
    )

    st.divider()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat = ChatHistory()
        st.rerun()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = ChatHistory()

if "config" not in st.session_state:
    st.session_state.config = None

if "chain" not in st.session_state:
    st.session_state.chain = None


# ─────────────────────────────────────────────
# LOAD CHAIN
# ─────────────────────────────────────────────
def load_chain(alpha, use_rerank, llm_provider):
    base_retriever = get_hybrid_retriever(alpha=alpha)

    if use_rerank:
        retriever = get_reranker(base_retriever)
    else:
        retriever = base_retriever

    return build_chain_with_sources(retriever, llm_provider)


# ─────────────────────────────────────────────
# CONFIG CHANGE HANDLING
# ─────────────────────────────────────────────
current_config = (alpha, use_rerank, llm_provider)

if st.session_state.config != current_config:
    st.session_state.config = current_config
    st.session_state.chain = None  # force reload


# ─────────────────────────────────────────────
# INIT CHAIN
# ─────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("🔧 Initializing system..."):
        st.session_state.chain = load_chain(alpha, use_rerank, llm_provider)

chain = st.session_state.chain


# ─────────────────────────────────────────────
# DISPLAY CHAT
# ─────────────────────────────────────────────
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {s.metadata.get('file_name','Unknown')} "
                        f"(Page {s.metadata.get('page','?')})**"
                    )
                    st.caption(s.page_content[:300] + "...")


# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────
if question := st.chat_input("Ask about SEBI regulations..."):

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching regulations..."):

            try:
                placeholder = st.empty()

                result = chain.invoke(question)
                full_answer = result.get("answer", "")

                streamed_text = ""

                # 🔥 Simulate streaming
                for chunk in full_answer.split():
                    streamed_text += chunk + " "
                    placeholder.markdown(streamed_text)
                    time.sleep(0.02)   # speed control

                answer = full_answer
               
                st.markdown(answer)

                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(
                                f"**[{i}] {s.metadata.get('file_name','Unknown')} "
                                f"(Page {s.metadata.get('page','?')})**"
                            )
                            st.caption(s.page_content[:300] + "...")

                # Save
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                st.session_state.chat.add(question, answer)

            except Exception as e:
                st.error(f"❌ Error: {e}")


# ─────────────────────────────────────────────
# DEBUG PANEL
# ─────────────────────────────────────────────
with st.expander("🔧 Debug Info"):
    st.write("Messages:", len(st.session_state.messages))
    st.write("Config:", st.session_state.config)