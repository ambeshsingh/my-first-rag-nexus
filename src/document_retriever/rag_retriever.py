import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root and src to sys.path for direct script execution
project_root = Path(__file__).resolve().parents[2]
src_dir = Path(__file__).resolve().parents[1]
for path in (str(project_root), str(src_dir)):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# ==============================================================================
# BLOCK: FlashRank Cross-Encoder Import
# Purpose: Imports FlashRank Ranker and RerankRequest for high-precision,
#          low-latency CPU cross-encoder reranking (Stage 2 retrieval).
# ==============================================================================
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker, RerankRequest = None, None

# ==============================================================================
# BLOCK: Dynamic EnsembleRetriever Import
# Purpose: Import EnsembleRetriever with fallback support across different
#          LangChain module structures (langchain vs langchain_classic).
# ==============================================================================
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.document_embedder.chroma_store import get_or_create_vector_store

# ==============================================================================
# BLOCK: Role-Based Access Control (RBAC) Permissions Hierarchy
# Purpose: Maps each user persona to the list of document categories they are
#          authorized to access. Ensures strict access boundaries.
# ==============================================================================
PERSONA_PERMISSIONS = {
    # General employee: general policies, benefits, travel, conduct, handbook, AI usage
    "employee": ["employee"],
    # Engineering: architecture, SDLC, project specs, plus security and general employee docs
    "engineering": ["engineering", "security", "employee"],
    # Human Resources: performance reviews, talent policies, plus general employee docs
    "hr": ["hr", "employee"],
    # Sales & Client Success: onboarding playbooks, client SOWs, plus general employee docs
    "sales": ["sales", "employee"],
    # Security & Compliance: incident response, information security, GDPR, plus general
    "security": ["security", "engineering", "employee"],
    # Finance & Operations: procurement and vendor policies, plus general
    "finance": ["finance", "employee"],
    # Executive & C-Suite: full unrestricted access to all company documents
    "executive": [
        "executive",
        "finance",
        "hr",
        "engineering",
        "security",
        "sales",
        "employee",
    ],
}


# ==============================================================================
# BLOCK: LLM Factory Configuration
# Purpose: Instantiates the Google Gemini model via Vertex AI or Gemini API.
#          Uses temperature=0.0 for deterministic, strictly grounded outputs.
# ==============================================================================
def get_llm(
    project_name: str = "poc-data-analytics-501104",
    model_name: str = "gemini-3.5-flash",
    location: str = "global",
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Returns the chat model for answer generation.
    Uses temperature=0.0 for strict factual consistency and reduced hallucination.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )

    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        project=project_name,
        model=model_name,
        location=location,
        temperature=temperature,
    )


# ==============================================================================
# BLOCK: Local Document Reconstruction from Chroma
# Purpose: Extracts all chunks directly from local ChromaDB storage using .get().
#          Avoids redownloading or re-chunking files from Google Cloud Storage.
# ==============================================================================
def get_all_documents_from_vector_store(vector_store: Chroma) -> List[Document]:
    """
    Fetches all chunk texts and metadata records from the local Chroma vector store.
    Converts raw Chroma storage dictionaries into standard LangChain Document objects.
    """
    raw_data = vector_store.get()
    documents = raw_data.get("documents", [])
    metadatas = raw_data.get("metadatas", [])

    all_docs = []
    for text, meta in zip(documents, metadatas):
        all_docs.append(Document(page_content=text, metadata=meta or {}))

    return all_docs


# ==============================================================================
# BLOCK: Persona-Filtered BM25 Sparse Keyword Retriever
# Purpose: Builds a BM25 index over ONLY the documents authorized for the user's
#          persona, ensuring keyword search cannot leak restricted content.
# ==============================================================================
def get_bm25_retriever(
    documents: List[Document],
    persona: Optional[str] = "employee",
    k: int = 5,
) -> BM25Retriever:
    """
    Constructs a BM25 keyword retriever restricted to the authorized persona subset.

    RBAC Enforcement:
    - Extracts permitted categories for the persona from PERSONA_PERMISSIONS.
    - Filters the document pool so unauthorized chunks are never indexed by BM25.
    - Returns a BM25Retriever configured with top-k retrieval.
    """
    allowed_categories = PERSONA_PERMISSIONS.get(
        persona.lower() if persona else "employee", ["employee"]
    )

    # Filter documents to only authorized categories
    authorized_docs = [
        doc for doc in documents
        if doc.metadata.get("persona") in allowed_categories
    ]

    # Graceful fallback if no documents match persona
    if not authorized_docs:
        authorized_docs = [
            Document(
                page_content="No authorized documents available for this role.",
                metadata={"persona": "none"},
            )
        ]

    bm25 = BM25Retriever.from_documents(authorized_docs)
    bm25.k = k
    return bm25


# ==============================================================================
# BLOCK: Dense Vector Retriever with MMR (Maximal Marginal Relevance)
# Purpose: Performs dense vector semantic search using Chroma MMR with pre-retrieval
#          metadata filtering for persona-based access control.
# ==============================================================================
def get_advanced_retriever(
    vector_store: Chroma,
    persona: Optional[str] = "employee",
    search_type: str = "mmr",
    k: int = 5,
    fetch_k: int = 25,
    lambda_mult: float = 0.7,
):
    """
    Creates an advanced MMR dense vector retriever with Pre-Retrieval Metadata Filtering.

    Why Pre-Retrieval Filtering?
    - Chroma filters out unauthorized documents BEFORE computing vector distances.
    - Guarantees forbidden documents never enter LLM context (immune to prompt injection).
    - Ensures all k=5 slots are filled with documents the user is actually permitted to view.
    """
    search_kwargs = {
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }

    if persona:
        persona_key = persona.lower()
        allowed = PERSONA_PERMISSIONS.get(persona_key, ["employee"])
        search_kwargs["filter"] = {"persona": {"$in": allowed}}

    if search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
    elif search_type == "similarity_score_threshold":
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.5},
        )
    return vector_store.as_retriever(search_kwargs=search_kwargs)


# ==============================================================================
# BLOCK: Hybrid Search Ensemble Retriever (Chroma MMR + BM25 with RRF)
# Purpose: Fuses dense semantic search and sparse lexical keyword search using
#          Reciprocal Rank Fusion (RRF) with configurable weighting.
# ==============================================================================
def get_hybrid_retriever(
    vector_store: Chroma,
    persona: Optional[str] = "employee",
    k: int = 5,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> EnsembleRetriever:
    """
    Creates a Hybrid Ensemble Retriever combining Chroma MMR (dense) and BM25 (sparse).

    How it works:
    1. Dense Leg: Runs Chroma MMR search with persona metadata filtering ($in operator).
    2. Sparse Leg: Runs BM25 search over the persona-filtered document subset.
    3. Fusion: Combines ranks using Reciprocal Rank Fusion (RRF) with weights:
       - 0.6 Dense: Prioritizes deep semantic and conceptual alignment.
       - 0.4 Sparse: Boosts exact keywords, project codenames, and acronyms.
    """
    # 1. Build dense retriever with Chroma MMR
    dense_retriever = get_advanced_retriever(
        vector_store=vector_store,
        persona=persona,
        search_type="mmr",
        k=k,
    )

    # 2. Extract local documents to build persona-filtered BM25 index
    all_docs = get_all_documents_from_vector_store(vector_store)
    sparse_retriever = get_bm25_retriever(
        documents=all_docs,
        persona=persona,
        k=k,
    )

    # 3. Create Ensemble Retriever using Reciprocal Rank Fusion (RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight],
    )
    return ensemble_retriever


# ==============================================================================
# BLOCK: FlashRank Cross-Encoder Model Singleton
# Purpose: Lazily initializes and caches the FlashRank model instance in memory
#          to prevent redundant disk I/O and ONNX session initialization on every query.
# ==============================================================================
_FLASHRANK_RANKER: Optional[object] = None


def get_cross_encoder_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Optional[object]:
    """
    Returns the cached FlashRank Ranker singleton.
    Uses 'ms-marco-TinyBERT-L-2-v2' (default) for ultra-fast, high-precision ONNX CPU inference.
    """
    global _FLASHRANK_RANKER
    if Ranker is None:
        return None
    if _FLASHRANK_RANKER is None:
        _FLASHRANK_RANKER = Ranker(model_name=model_name)
    return _FLASHRANK_RANKER


# ==============================================================================
# BLOCK: Cross-Encoder Document Reranker (Stage 2)
# Purpose: Takes high-recall candidates from Stage 1 (Hybrid Search) and re-evaluates
#          every (query, document) pair with full cross-attention to score true relevance.
# ==============================================================================
def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    model_name: str = "ms-marco-TinyBERT-L-2-v2",
) -> List[Document]:
    """
    Re-scores and re-ranks retrieved chunks using the FlashRank Cross-Encoder.

    Parameters:
    - query: The original user question.
    - documents: The candidate chunks retrieved from Stage 1 (e.g. 15-20 candidates).
    - top_k: Number of highest-relevance chunks to return for LLM synthesis (default 5).

    How it works:
    1. Packages query + chunk texts into FlashRank RerankRequest passages.
    2. Executes ONNX cross-attention scoring between query tokens and document tokens.
    3. Attaches 'rerank_score' and 'stage1_rank' to each document's metadata.
    4. Truncates and returns the top_k most relevant chunks.
    """
    if not documents:
        return []

    ranker = get_cross_encoder_ranker(model_name=model_name)
    if ranker is None:
        return documents[:top_k]

    # Convert LangChain Documents to FlashRank passage dicts
    passages = [
        {
            "id": idx,
            "text": doc.page_content,
            "meta": doc.metadata,
        }
        for idx, doc in enumerate(documents)
    ]

    request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(request)

    # Reconstruct top_k Document objects with rerank scores and rank tracking
    reranked_docs = []
    for new_rank, item in enumerate(reranked_results[:top_k], start=1):
        original_idx = item["id"]
        original_doc = documents[original_idx]

        # Preserve all original metadata and record rerank insights
        new_metadata = dict(original_doc.metadata)
        new_metadata["rerank_score"] = float(item["score"])
        new_metadata["stage1_rank"] = original_idx + 1  # 1-based index in Stage 1
        new_metadata["stage2_rank"] = new_rank

        reranked_docs.append(
            Document(
                page_content=original_doc.page_content,
                metadata=new_metadata,
            )
        )

    return reranked_docs


# ==============================================================================
# BLOCK: Document Context Formatter with Citations
# Purpose: Formats retrieved chunks into clean markdown blocks labeled with
#          source filename, page number, persona access level, and rerank score.
# ==============================================================================
def format_documents_with_citations(docs: List[Document]) -> str:
    """
    Formats retrieved document chunks into clean, cited context blocks for the LLM.
    Includes the cross-encoder score if available.
    """
    if not docs:
        return "No relevant context found in documents."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        role = doc.metadata.get("persona", "all").upper()
        score_info = ""
        if "rerank_score" in doc.metadata:
            score_info = f" | Score: {doc.metadata['rerank_score']:.4f}"
        header = f"[Excerpt #{idx} | Role: {role} | Source: {source} | Page: {page}{score_info}]"
        content = doc.page_content.strip()
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted_blocks)


# ==============================================================================
# BLOCK: RAG Answer Synthesis Chain
# Purpose: Builds the prompt template and binds the Gemini LLM with output parser.
#          Enforces grounding, citation rules, and anti-hallucination behavior.
# ==============================================================================
def build_rag_chain(llm: Optional[BaseChatModel] = None):
    """
    Builds the core prompt + LLM answer synthesis chain.
    Expects a dictionary with {"context": str, "question": str}.
    """
    llm = llm or get_llm()

    system_prompt = (
        "You are an expert enterprise research assistant for Nexus Company.\n"
        "Your task is to answer the user's question accurately, concisely, and completely "
        "using ONLY the provided document context excerpts below.\n\n"
        "Grounding Rules:\n"
        "1. Base your answer strictly on facts stated in the context. Never fabricate or extrapolate.\n"
        "2. If the context does not contain enough information to answer, state clearly: "
        "'Based on the available documents, I do not have enough information to answer this question.'\n"
        "3. Provide specific citations for your statements using the excerpt source and page number "
        "(e.g., '[Source: filename.pdf, Page: X]').\n\n"
        "Context Excerpts:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


# ==============================================================================
# BLOCK: Contextual Query Reformulation (Multi-Turn Conversational Memory)
# Purpose: Uses chat history to resolve pronouns ("it", "that project", "those terms")
#          into a standalone search query optimized for vector and keyword retrieval.
# ==============================================================================
def reformulate_query(
    question: str,
    chat_history: Optional[List[dict]] = None,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Rewrites a conversational follow-up question into an independent, standalone
    search query using prior turns from chat_history.

    If chat_history is empty or None, returns the original question unchanged.
    """
    if not chat_history:
        return question

    llm = llm or get_llm()

    # Format the recent history turns (last 6 messages max)
    history_lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        # Avoid including diagnostic blocks in history
        if "🔍 Retrieval & Rerank Diagnostics" in content:
            content = content.split("🔍 Retrieval & Rerank Diagnostics")[0].strip()
        if content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return question

    history_str = "\n".join(history_lines)

    prompt = (
        "Given the conversation history and a follow-up question below, "
        "rephrase the follow-up question to be a standalone, self-contained search query.\n\n"
        "Guidelines:\n"
        "1. Resolve all pronouns (e.g., 'it', 'its', 'they', 'that policy', 'this project') to the exact entity mentioned in history.\n"
        "2. Keep company names, project names, document types, or technical terms intact.\n"
        "3. Do NOT answer the question. Only output the rewritten search query.\n"
        "4. If the question is already fully self-contained, return it verbatim.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Follow-up Question: {question}\n\n"
        "Standalone Search Query:"
    )

    response = llm.invoke(prompt)
    standalone = response.content if hasattr(response, "content") else str(response)

    # Safely handle list of parts from multimodal or Vertex chat output formats
    if isinstance(standalone, list):
        text_parts = []
        for part in standalone:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
            else:
                text_parts.append(str(part))
        standalone = " ".join(text_parts)
    elif not isinstance(standalone, str):
        standalone = str(standalone)

    cleaned = standalone.strip().strip('"').strip("'")
    return cleaned or question


# ==============================================================================
# BLOCK: Multi-Turn Conversational Query Pipeline with Diagnostics
# Purpose: Orchestrates multi-turn conversational query execution, running:
#          1. Contextual Query Reformulation (Chat Memory)
#          2. Two-Stage Retrieve & Rerank (RBAC Pre-filtering + FlashRank)
#          3. Grounded Answer Synthesis
#          Returns both the synthesized answer and rich diagnostic metadata for UI inspection.
# ==============================================================================
def ask_question_conversational(
    vector_store: Chroma,
    question: str,
    chat_history: Optional[List[dict]] = None,
    persona: str = "employee",
    search_mode: str = "rerank",
    k: int = 5,
    initial_k: int = 15,
) -> dict:
    """
    Executes a conversational query:
    1. Reformulates ambiguous follow-up questions using chat_history.
    2. Runs Stage 1 (Hybrid or MMR) with Pre-Retrieval RBAC metadata filtering.
    3. Runs Stage 2 (FlashRank Cross-Encoder reranking) if search_mode=='rerank'.
    4. Formats context excerpts and calls Gemini with strict grounding.
    5. Returns dictionary with answer, standalone_query, retrieved_docs, initial_candidates.
    """
    # 1. Reformulate question if prior conversation turns exist
    standalone_query = reformulate_query(question=question, chat_history=chat_history)

    # 2. Execute retrieval based on search_mode using standalone_query
    initial_candidates = []
    if search_mode == "rerank":
        # STAGE 1: Broad Hybrid Search (k=initial_k)
        hybrid_retriever = get_hybrid_retriever(vector_store, persona=persona, k=initial_k)
        initial_candidates = hybrid_retriever.invoke(standalone_query)

        # STAGE 2: FlashRank Cross-Encoder Reranker
        retrieved_docs = rerank_documents(
            query=standalone_query,
            documents=initial_candidates,
            top_k=k,
        )
    elif search_mode == "hybrid":
        retriever = get_hybrid_retriever(vector_store, persona=persona, k=k)
        retrieved_docs = retriever.invoke(standalone_query)
        initial_candidates = retrieved_docs
    else:
        retriever = get_advanced_retriever(vector_store, persona=persona, search_type="mmr", k=k)
        retrieved_docs = retriever.invoke(standalone_query)
        initial_candidates = retrieved_docs

    # 3. Format context with citations and generate answer
    context_text = format_documents_with_citations(retrieved_docs)
    chain = build_rag_chain()
    answer = chain.invoke({"context": context_text, "question": question})

    return {
        "answer": answer,
        "standalone_query": standalone_query,
        "retrieved_docs": retrieved_docs,
        "initial_candidates": initial_candidates,
        "persona": persona,
        "search_mode": search_mode,
    }


# ==============================================================================
# BLOCK: Query Execution and Response Pipeline (CLI Wrapper)
# Purpose: Handles CLI terminal execution while delegating core logic to the
#          conversational engine. Prints formatted diagnostics to stdout.
# ==============================================================================
def ask_question(
    vector_store: Chroma,
    question: str,
    persona: str = "employee",
    search_mode: str = "rerank",
    k: int = 5,
    initial_k: int = 15,
) -> str:
    """
    Executes a query through the RAG pipeline for CLI usage:
    - search_mode='rerank' (Default): Two-Stage Retrieve & Rerank.
    - search_mode='hybrid': Dense MMR + Sparse BM25 via Reciprocal Rank Fusion.
    - search_mode='mmr': Pure Dense Vector MMR search.
    """
    print("\n" + "=" * 70)
    print(f"USER PERSONA : {persona.upper()}")
    print(f"SEARCH MODE  : {search_mode.upper()}")
    print(f"QUESTION     : {question}")
    print("=" * 70)

    result = ask_question_conversational(
        vector_store=vector_store,
        question=question,
        chat_history=None,
        persona=persona,
        search_mode=search_mode,
        k=k,
        initial_k=initial_k,
    )

    if result["standalone_query"] != question:
        print(f"\n[Multi-Turn Contextualizer] Reformulated Query: '{result['standalone_query']}'")

    if search_mode == "rerank":
        print(f"\n[Stage 1: Hybrid Retrieval] Fetched {len(result['initial_candidates'])} candidate chunk(s):")
        for idx, doc in enumerate(result["initial_candidates"], start=1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "N/A")
            role_tag = doc.metadata.get("persona", "all").upper()
            print(f"  Stage-1 #{idx:2d} | [{role_tag:11}] {source} (Page {page})")

        print(f"\n[Stage 2: Cross-Encoder Rerank] Selected Top-{len(result['retrieved_docs'])} Reranked chunk(s):")
        for idx, doc in enumerate(result["retrieved_docs"], start=1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "N/A")
            role_tag = doc.metadata.get("persona", "all").upper()
            score = doc.metadata.get("rerank_score", 0.0)
            orig_rank = doc.metadata.get("stage1_rank", "N/A")
            print(f"  Rank #{idx} (was #{orig_rank:2d} in Stg 1) | Score: {score:.5f} | [{role_tag:11}] {source} (p. {page})")
    else:
        print(f"\nRetrieved {len(result['retrieved_docs'])} authorized chunk(s):")
        for idx, doc in enumerate(result["retrieved_docs"], start=1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "N/A")
            role_tag = doc.metadata.get("persona", "all").upper()
            print(f"  {idx}. [{role_tag:11}] {source} (Page {page}) - {len(doc.page_content)} chars")

    print("\nGenerating grounded response with citations...")
    print(f"\nANSWER:\n{result['answer']}")
    print("=" * 70 + "\n")
    return result["answer"]


# ==============================================================================
# BLOCK: CLI Entrypoint
# Purpose: Allows testing queries directly from the terminal with custom persona
#          and search mode flags.
#          Usage: python src/document_retriever/rag_retriever.py "query" --persona engineering --mode rerank
# ==============================================================================
if __name__ == "__main__":
    # 1. Load the persistent Chroma vector store
    store = get_or_create_vector_store()

    # 2. Parse command-line arguments: question, --persona <role>, and --mode <rerank|hybrid|mmr>
    persona_arg = "employee"
    mode_arg = "rerank"
    args = sys.argv[1:]

    # Parse persona flag
    if "--persona" in args:
        idx = args.index("--persona")
        if idx + 1 < len(args):
            persona_arg = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    # Parse mode flag
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode_arg = args[idx + 1].lower()
            args = args[:idx] + args[idx + 2 :]

    default_query = "What are the objectives of Project Aegis?"
    query = args[0] if args and not args[0].startswith("--") else default_query

    ask_question(store, question=query, persona=persona_arg, search_mode=mode_arg, k=5)
