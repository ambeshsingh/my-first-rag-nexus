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
# BLOCK: Document Context Formatter with Citations
# Purpose: Formats retrieved chunks into clean markdown blocks labeled with
#          source filename, page number, and persona access level.
# ==============================================================================
def format_documents_with_citations(docs: List[Document]) -> str:
    """
    Formats retrieved document chunks into clean, cited context blocks for the LLM.
    """
    if not docs:
        return "No relevant context found in documents."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        role = doc.metadata.get("persona", "all").upper()
        header = f"[Excerpt #{idx} | Role: {role} | Source: {source} | Page: {page}]"
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
# BLOCK: Query Execution and Response Pipeline
# Purpose: Handles the complete end-to-end question answering pipeline:
#          retrieval (Hybrid or MMR) -> citation printing -> LLM synthesis.
# ==============================================================================
def ask_question(
    vector_store: Chroma,
    question: str,
    persona: str = "employee",
    search_mode: str = "hybrid",
    k: int = 5,
) -> str:
    """
    Executes a query through the RAG pipeline:
    - search_mode='hybrid': Fuses Dense MMR and Sparse BM25 via Reciprocal Rank Fusion.
    - search_mode='mmr': Uses pure Dense Vector MMR search.
    """
    # Select retriever based on search_mode
    if search_mode == "hybrid":
        retriever = get_hybrid_retriever(vector_store, persona=persona, k=k)
    else:
        retriever = get_advanced_retriever(vector_store, persona=persona, search_type="mmr", k=k)

    chain = build_rag_chain()

    # --- SINGLE RETRIEVAL CALL WITH PRE-FILTERING ---
    retrieved_docs = retriever.invoke(question)

    print("\n" + "=" * 65)
    print(f"USER PERSONA : {persona.upper()}")
    print(f"SEARCH MODE  : {search_mode.upper()}")
    print(f"QUESTION     : {question}")
    print("=" * 65)
    print(f"Retrieved {len(retrieved_docs)} authorized chunk(s):")
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        role_tag = doc.metadata.get("persona", "all").upper()
        print(f"  {idx}. [{role_tag:11}] {source} (Page {page}) - {len(doc.page_content)} chars")

    # Format the retrieved docs as context
    context_text = format_documents_with_citations(retrieved_docs)

    print("\nGenerating grounded response with citations...")
    answer = chain.invoke({"context": context_text, "question": question})

    print(f"\nANSWER:\n{answer}")
    print("=" * 65 + "\n")
    return answer


# ==============================================================================
# BLOCK: CLI Entrypoint
# Purpose: Allows testing queries directly from the terminal with custom persona
#          and search mode flags.
#          Usage: python src/document_retriever/rag_retriever.py "query" --persona engineering --mode hybrid
# ==============================================================================
if __name__ == "__main__":
    # 1. Load the persistent Chroma vector store
    store = get_or_create_vector_store()

    # 2. Parse command-line arguments: question, --persona <role>, and --mode <hybrid|mmr>
    persona_arg = "employee"
    mode_arg = "hybrid"
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
