import json
import sys
from pathlib import Path
from google.cloud import storage

# Add project root and src to sys.path for direct script execution
project_root = Path(__file__).resolve().parents[2]
src_dir = Path(__file__).resolve().parents[1]
for path in (str(project_root), str(src_dir)):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader.gcs_file_loader import gcs_file_loader


def recursive_text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs


def upload_chunks_to_gcs(
    chunks,
    bucket_name: str,
    destination_blob_name: str,
    project_name: str | None = None,
) -> None:
    """
    Serializes LangChain Document chunks into JSON Lines (JSONL) format
    and uploads them directly to Google Cloud Storage.
    """
    client = storage.Client(project=project_name) if project_name else storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Format each chunk as a structured JSON object
    lines = []
    for idx, doc in enumerate(chunks):
        record = {
            "chunk_id": f"chunk_{idx:05d}",
            "text": doc.page_content,
            "metadata": doc.metadata,
            "doc_title": doc.metadata["source"].split("/")[-1],
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    jsonl_content = "\n".join(lines)

    # Upload string content to GCS
    blob.upload_from_string(jsonl_content, content_type="application/jsonl")
    print(f"Uploaded {len(chunks)} chunks to gs://{bucket_name}/{destination_blob_name}")


if __name__ == "__main__":
    # 1. Load documents from GCS
    raw_docs = gcs_file_loader()

    # 2. Split into chunks
    chunks = recursive_text_splitter(raw_docs)
    print(f"Total chunks generated: {len(chunks)}")

    # 3. Preview first chunk structure
    if chunks:
        sample_record = {
            "chunk_id": "chunk_00000",
            "text": chunks[0].page_content[:150] + "...",
            "metadata": chunks[0].metadata,
        }
        print("\n--- Example Stored Chunk (JSON Schema) ---")
        print(json.dumps(sample_record, indent=2))

    # 4. Upload chunks to GCS in JSON Lines format
    upload_chunks_to_gcs(
        chunks=chunks,
        bucket_name="amby-open-ragbench-bucket",
        destination_blob_name="nexus_chunks/chunks.jsonl",
        project_name="poc-data-analytics-501104",
    )

    # for chunk in chunks:
    #     print(chunk.metadata)
    #     break

