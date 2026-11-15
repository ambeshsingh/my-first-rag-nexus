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
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import VertexAIEmbeddings

from src.document_chunker.recursive_text_splitter import recursive_text_splitter
from src.document_loader.gcs_file_loader import gcs_file_loader

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "nexus_rag_docs"


def get_embedding_model(
    project_name: str = "poc-data-analytics-501104",
    model_name: str = "text-embedding-005",
) -> Embeddings:
    """Returns the VertexAI embedding model."""
    return VertexAIEmbeddings(project=project_name, model=model_name)


def load_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Loads an existing persistent Chroma vector store from disk."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function or get_embedding_model(),
        persist_directory=persist_directory,
    )


def create_vector_store(
    chunks: List[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Embeds document chunks and stores them in a persistent Chroma vector database."""
    ids = [f"chunk_{idx:05d}" for idx in range(len(chunks))]
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function or get_embedding_model(),
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}' at '{persist_directory}'.")
    return vector_store


def get_or_create_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    force_reindex: bool = False,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """
    Returns the vector store. Loads directly from disk if already cached.
    Only triggers the one-time GCS load, chunking, and embedding if missing or if force_reindex=True.
    """
    embedding_function = embedding_function or get_embedding_model()

    if not force_reindex and Path(persist_directory).exists():
        vector_store = load_vector_store(persist_directory, collection_name, embedding_function)
        count = vector_store._collection.count()
        if count > 0:
            print(f"Loaded existing Chroma store ({count} chunks) from '{persist_directory}' (skipped GCS download & chunking).")
            return vector_store

    print("Building vector store: Loading from GCS -> Chunking -> Embedding...")
    raw_docs = gcs_file_loader()
    chunks = recursive_text_splitter(raw_docs)
    return create_vector_store(chunks, persist_directory, collection_name, embedding_function)


def query_similar_chunks(
    vector_store: Chroma,
    query: str,
    k: int = 5,
    persona: Optional[str] = None,
    use_mmr: bool = True,
    fetch_k: int = 25,
    lambda_mult: float = 0.6,
) -> List[Document]:
    """
    Queries the vector store for the most relevant chunks with optional persona filtering.
    
    Why MMR (Maximal Marginal Relevance)?
    - Standard similarity search often returns multiple redundant chunks from the same page.
    - MMR pulls a candidate pool of `fetch_k=25` chunks, then selects `k=5` chunks that
      balance similarity to the query with diversity from already-chosen chunks.
    """
    filter_dict = None
    if persona:
        from src.document_retriever.rag_retriever import PERSONA_PERMISSIONS
        allowed = PERSONA_PERMISSIONS.get(persona.lower(), ["employee"])
        filter_dict = {"persona": {"$in": allowed}}
        print(f"Pre-retrieval filter applied for persona '{persona}': {allowed}")

    if use_mmr:
        print(f"\nQuerying with MMR (k={k}, fetch_k={fetch_k}, lambda={lambda_mult}): '{query}'")
        docs = vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter_dict
        )
    else:
        print(f"\nQuerying with standard similarity search (k={k}): '{query}'")
        docs = vector_store.similarity_search(query, k=k, filter=filter_dict)

    for rank, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        role = doc.metadata.get("persona", "all").upper()
        print(f"\n--- Result #{rank} | [{role}] {source} | Page: {page} ---")
        print(f"{doc.page_content.strip()}\n")

    return docs


if __name__ == "__main__":
    should_reindex = "--reindex" in sys.argv

    vector_store = get_or_create_vector_store(force_reindex=should_reindex)

    test_query = "how much is performance bonus?"
    print("\n--- Testing as standard EMPLOYEE ---")
    query_similar_chunks(vector_store, query=test_query, k=5, persona="employee")

    print("\n--- Testing as EXECUTIVE ---")
    query_similar_chunks(vector_store, query=test_query, k=5, persona="executive")


