import os
import sys
from pathlib import Path

# Add project root and src to Python path for seamless module discovery
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"
for p in (str(project_root), str(src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from src.document_embedder.chroma_store import get_or_create_vector_store
from src.document_retriever.rag_retriever import (
    PERSONA_PERMISSIONS,
    ask_question_conversational,
)

# ==============================================================================
# BLOCK: Streamlit Page Configuration
# Purpose: Sets up page layout, browser title, and wide-screen responsiveness.
# ==============================================================================
st.set_page_config(
    page_title="Nexus Enterprise RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# BLOCK: Vector Store Resource Cache
# Purpose: Loads the persistent ChromaDB vector store once into Streamlit memory
#          using @st.cache_resource to prevent reloading SQLite/HNSW on every rerun.
# ==============================================================================
@st.cache_resource(show_spinner="Connecting to persistent vector store...")
def load_cached_vector_store():
    """Loads and caches the persistent Chroma vector database."""
    return get_or_create_vector_store()


# ==============================================================================
# BLOCK: Session State Initialization
# Purpose: Maintains conversational history, active persona, and diagnostic state
#          across user interactions.
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_persona" not in st.session_state:
    st.session_state.current_persona = "employee"


# ==============================================================================
# BLOCK: Sidebar Controls (RBAC Persona, Search Mode, and Parameters)
# Purpose: Provides an interactive control panel for switching user access levels,
#          selecting retrieval strategies, and tuning retrieval depth.
# ==============================================================================
with st.sidebar:
    st.title("🏢 Nexus RAG Controls")
    st.markdown("---")

    # 1. Role-Based Access Control (RBAC) Persona Selector
    st.subheader("👤 User Persona (RBAC)")
    selected_persona = st.selectbox(
        "Active Role:",
        options=list(PERSONA_PERMISSIONS.keys()),
        index=list(PERSONA_PERMISSIONS.keys()).index(st.session_state.current_persona),
        help="Select user identity to enforce Pre-Retrieval Metadata Filtering.",
    )

    # Detect persona switch to prevent cross-persona context leakage
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
        st.session_state.messages = []
        st.toast(f"Switched persona to: {selected_persona.upper()}. Chat history reset for RBAC security.", icon="🔒")

    # Display permitted document categories for the chosen persona
    allowed_categories = PERSONA_PERMISSIONS.get(selected_persona, [])
    st.markdown(
        "**Permitted Document Scope:**\n"
        + "".join([f"- `{cat}`\n" for cat in allowed_categories])
    )

    st.markdown("---")

    # 2. Retrieval Strategy Selector
    st.subheader("🔍 Retrieval Strategy")
    mode_options = {
        "Two-Stage Rerank (FlashRank)": "rerank",
        "Hybrid Search (Dense MMR + BM25)": "hybrid",
        "Dense MMR Only": "mmr",
    }
    selected_mode_label = st.selectbox(
        "Search Algorithm:",
        options=list(mode_options.keys()),
        index=0,
        help="Choose between Two-Stage Retrieve & Rerank, Hybrid RRF, or Pure Dense Vector search.",
    )
    search_mode = mode_options[selected_mode_label]

    st.markdown("---")

    # 3. Retrieval Hyperparameters
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider(
        "Final Context Chunks (Top-K):",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top chunks passed to the LLM for grounded answer generation.",
    )

    initial_k = 15
    if search_mode == "rerank":
        initial_k = st.slider(
            "Stage 1 Candidate Pool:",
            min_value=5,
            max_value=30,
            value=15,
            help="High-recall candidate pool retrieved in Stage 1 before Cross-Encoder reranking.",
        )

    # 4. Clear Conversation Button
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Nexus Enterprise RAG v3.0 • Powered by Gemini & FlashRank")


# ==============================================================================
# BLOCK: Main Chat Header & System Banner
# Purpose: Displays current system configuration, persona scope, and status.
# ==============================================================================
st.title("💬 Nexus Enterprise Research Assistant")
st.markdown(
    f"Currently chatting as **`{selected_persona.upper()}`** using "
    f"**`{selected_mode_label}`** | Strict Document Grounding Enforced"
)

# Load the vector database
try:
    vector_store = load_cached_vector_store()
except Exception as e:
    st.error(f"Failed to load Chroma vector database: {e}")
    st.stop()


# ==============================================================================
# BLOCK: Conversation History Renderer with Diagnostic Inspection
# Purpose: Renders previous user and assistant turns with expandable diagnostic
#          drawers showing reformulated queries, candidate rankings, and scores.
# ==============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render diagnostic drawer for assistant responses
        if msg["role"] == "assistant" and "diagnostics" in msg:
            diag = msg["diagnostics"]
            with st.expander("🔍 View Retrieval & Rerank Diagnostics", expanded=False):
                # 1. Multi-Turn Reformulation Insight
                if diag.get("standalone_query") and diag.get("standalone_query") != diag.get("original_query"):
                    st.markdown("**🔄 Contextual Query Reformulation:**")
                    st.info(
                        f"**Original Follow-up:** {diag.get('original_query')}\n\n"
                        f"**Rewritten Search Query:** `{diag.get('standalone_query')}`"
                    )

                # 2. Retrieved Chunks and Rerank Breakdown
                retrieved_docs = diag.get("retrieved_docs", [])
                st.markdown(f"**Retrieved Chunks ({len(retrieved_docs)}):**")

                for rank, doc in enumerate(retrieved_docs, start=1):
                    source = Path(doc.metadata.get("source", "Unknown")).name
                    page = doc.metadata.get("page", "N/A")
                    role = doc.metadata.get("persona", "all").upper()
                    score = doc.metadata.get("rerank_score")
                    stg1_rank = doc.metadata.get("stage1_rank")

                    score_str = f" • FlashRank Score: **{score:.4f}**" if score is not None else ""
                    rank_shift = f" (was Stage-1 #{stg1_rank})" if stg1_rank is not None else ""

                    st.markdown(
                        f"**#{rank}{rank_shift}** • `[{role}]` `{source}` (Page {page}){score_str}"
                    )
                    with st.container():
                        st.text(doc.page_content.strip()[:400] + ("..." if len(doc.page_content) > 400 else ""))
                        st.divider()


# ==============================================================================
# BLOCK: User Input & Multi-Turn Response Execution
# Purpose: Handles new user chat submissions, runs conversational reformulation,
#          executes two-stage retrieval, synthesizes answer, and updates session.
# ==============================================================================
if user_query := st.chat_input("Ask a question about Nexus policies, architecture, or projects..."):
    # 1. Display user query immediately
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Execute Conversational RAG with spinner
    with st.chat_message("assistant"):
        with st.spinner("Retrieving authorized documents and synthesizing answer..."):
            try:
                # Prepare history for contextualizer (exclude current prompt)
                history_for_llm = st.session_state.messages[:-1]

                rag_result = ask_question_conversational(
                    vector_store=vector_store,
                    question=user_query,
                    chat_history=history_for_llm,
                    persona=selected_persona,
                    search_mode=search_mode,
                    k=top_k,
                    initial_k=initial_k,
                )

                answer = rag_result["answer"]
                st.markdown(answer)

                # Render diagnostic accordion
                diagnostics = {
                    "original_query": user_query,
                    "standalone_query": rag_result.get("standalone_query"),
                    "retrieved_docs": rag_result.get("retrieved_docs", []),
                    "initial_candidates": rag_result.get("initial_candidates", []),
                }

                with st.expander("🔍 View Retrieval & Rerank Diagnostics", expanded=False):
                    if diagnostics["standalone_query"] and diagnostics["standalone_query"] != user_query:
                        st.markdown("**🔄 Contextual Query Reformulation:**")
                        st.info(
                            f"**Original Follow-up:** {user_query}\n\n"
                            f"**Rewritten Search Query:** `{diagnostics['standalone_query']}`"
                        )

                    st.markdown(f"**Retrieved Chunks ({len(diagnostics['retrieved_docs'])}):**")
                    for rank, doc in enumerate(diagnostics["retrieved_docs"], start=1):
                        source = Path(doc.metadata.get("source", "Unknown")).name
                        page = doc.metadata.get("page", "N/A")
                        role = doc.metadata.get("persona", "all").upper()
                        score = doc.metadata.get("rerank_score")
                        stg1_rank = doc.metadata.get("stage1_rank")

                        score_str = f" • FlashRank Score: **{score:.4f}**" if score is not None else ""
                        rank_shift = f" (was Stage-1 #{stg1_rank})" if stg1_rank is not None else ""

                        st.markdown(
                            f"**#{rank}{rank_shift}** • `[{role}]` `{source}` (Page {page}){score_str}"
                        )
                        with st.container():
                            st.text(doc.page_content.strip()[:400] + ("..." if len(doc.page_content) > 400 else ""))
                            st.divider()

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "diagnostics": diagnostics,
                })

            except Exception as ex:
                st.error(f"An error occurred during query execution: {ex}")

