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
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.document_embedder.chroma_store import get_or_create_vector_store

# Role-Based Access Control (RBAC) permission hierarchy:
# Each persona can only access documents tagged with their permitted categories.
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


def get_advanced_retriever(
    vector_store: Chroma,
    persona: Optional[str] = "employee",
    search_type: str = "mmr",
    k: int = 5,
    fetch_k: int = 25,
    lambda_mult: float = 0.7,
):
    """
    Creates an advanced MMR retriever with Pre-Retrieval Metadata Filtering.

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
        print(f"Pre-Retrieval Filter active for persona '{persona.upper()}': allowed_categories={allowed}")

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


def ask_question(
    vector_store: Chroma,
    question: str,
    persona: str = "employee",
    k: int = 5,
) -> str:
    """
    Executes a query with Pre-Retrieval Metadata Filtering:
    1. Filters Chroma vector search to ONLY chunks permitted for the given persona.
    2. Displays the source citations, pages, and persona tags.
    3. Formats and sends the context to the LLM for grounded answer generation.
    """
    retriever = get_advanced_retriever(vector_store, persona=persona, search_type="mmr", k=k)
    chain = build_rag_chain()

    # --- SINGLE RETRIEVAL CALL WITH PRE-FILTERING ---
    retrieved_docs = retriever.invoke(question)

    print("\n" + "=" * 65)
    print(f"USER PERSONA : {persona.upper()}")
    print(f"QUESTION     : {question}")
    print("=" * 65)
    print(f"Retrieved {len(retrieved_docs)} authorized chunk(s) via MMR:")
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


if __name__ == "__main__":
    # 1. Load the persistent Chroma vector store
    store = get_or_create_vector_store()

    # 2. Parse command-line arguments: question and --persona <role>
    # Example: python src/document_retriever/rag_retriever.py "how much is bonus?" --persona executive
    persona_arg = "employee"
    args = sys.argv[1:]
    if "--persona" in args:
        idx = args.index("--persona")
        if idx + 1 < len(args):
            persona_arg = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    default_query = "how much is performance bonus?"
    query = args[0] if args and not args[0].startswith("--") else default_query

    ask_question(store, question=query, persona=persona_arg, k=5)
