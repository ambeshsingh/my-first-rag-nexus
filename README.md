# Nexus Company Enterprise RAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Cloud Storage**, **ChromaDB**, and **Google Vertex AI / Gemini**.

This application implements **Role-Based Access Control (RBAC)** using **Pre-Retrieval Metadata Filtering** to ensure sensitive enterprise documents (e.g., executive bonuses, client SOWs, engineering specs) are strictly isolated based on user personas.

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Storage Layer"]
        GCS["Google Cloud Storage (PDFs)"] --> Loader["GCS File Loader (PyPDFLoader)"]
        Loader --> Tagger["Metadata Persona Tagger"]
        Tagger --> Splitter["Recursive Character Text Splitter"]
        Splitter --> Embed["Vertex AI Embeddings (text-embedding-005)"]
        Embed --> Chroma[("ChromaDB (./chroma_db)")]
    end

    subgraph Retrieval["2. Two-Stage Retrieve & Rerank"]
        User["User Question + Persona"] --> PreFilter["Persona RBAC Filter<br/>(Allowed Categories)"]
        
        subgraph Stage1["Stage 1: High-Recall Hybrid Retrieval (k=15)"]
            PreFilter --> Dense["Dense Leg: Chroma MMR (60%)"]
            PreFilter --> Sparse["Sparse Leg: BM25 Lexical (40%)"]
            Dense & Sparse --> RRF["Reciprocal Rank Fusion (RRF)"]
        end

        subgraph Stage2["Stage 2: High-Precision Cross-Encoder (FlashRank)"]
            RRF --> CrossEncoder["FlashRank ms-marco-TinyBERT<br/>(Full Query-Doc Cross-Attention)"]
            CrossEncoder --> RerankTop["Top K=5 Rescored Chunks"]
        end

        RerankTop --> Context["Formatted Excerpts with Citations & Scores"]
        Context --> LLM["Google Gemini (gemini-3.5-flash)"]
        LLM --> Output["Grounded Answer with Page Footnotes"]
    end
```

---

## 🔒 Role-Based Access Control (RBAC)

Documents are tagged at ingestion with persona categories. Chroma filters out unauthorized chunks **before** calculating vector similarity, guaranteeing forbidden documents never enter the LLM's context window.

### Persona Permission Matrix

| User Persona | Permitted Document Categories | Example Accessible Documents |
| :--- | :--- | :--- |
| **`employee`** | `employee` | Employee Handbook, Code of Conduct, Benefits Guide, AI Usage Policy, Travel Policy |
| **`engineering`** | `engineering`, `security`, `employee` | Cloud Architecture Standards, SDLC Process, Project Specs, Incident Playbook |
| **`hr`** | `hr`, `employee` | Performance Review Policy, Employee Relations |
| **`sales`** | `sales`, `employee` | Client Onboarding Playbook, Project Client SOW |
| **`security`** | `security`, `engineering`, `employee` | InfoSec Policy, Incident Response, GDPR & Data Privacy |
| **`finance`** | `finance`, `employee` | Procurement & Vendor Policy |
| **`executive`** | **ALL** categories (unrestricted) | Compensation & Bonus Structure, Strategic Product Roadmaps, SOWs, all policies |

---

## 📁 Repository Structure

```
nexus_company_rag/
├── src/
│   ├── document_loader/          # Ingestion from GCS & persona tagging
│   │   ├── __init__.py
│   │   └── gcs_file_loader.py
│   ├── document_chunker/         # Text splitting & JSONL upload
│   │   ├── __init__.py
│   │   └── recursive_text_splitter.py
│   ├── document_embedder/        # Chroma vector database & persistence
│   │   ├── __init__.py
│   │   └── chroma_store.py
│   └── document_retriever/       # MMR retriever & grounded LLM generation
│       ├── __init__.py
│       └── rag_retriever.py
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore file
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Cloud SDK (`gcloud`) authenticated:
  ```bash
  gcloud auth application-default login
  gcloud config set project poc-data-analytics-501104
  ```

### 2. Installation
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```

---

## 💻 Usage

### 1. Build / Re-index the Vector Database
Download PDFs from GCS, attach persona metadata, chunk, and persist to `./chroma_db`:
```bash
python src/document_embedder/chroma_store.py --reindex
```

### 2. Launch the Interactive Streamlit Web UI

Start the multi-turn conversational chat interface with live persona switching and retrieval diagnostics:
```bash
streamlit run app.py
```
- **Persona Switcher:** Switch between `Employee`, `Engineering`, `HR`, `Sales`, `Security`, `Finance`, and `Executive` in the sidebar.
- **Conversational Memory:** Resolves conversational follow-ups (e.g., *"Who leads that?"* $\to$ *"Who leads Project Aegis?"*).
- **Retrieval & Rerank Inspector:** Click the expandable drawer beneath each response to view the reformulated search query, Stage 1 candidate ranks, and Stage 2 FlashRank cross-encoder scores.

---

### 3. CLI Querying with Two-Stage Cross-Encoder Reranking

#### Query Technical Specs / Project Codename (Retrieve & Rerank):
```bash
python src/document_retriever/rag_retriever.py "What are the objectives of Project Aegis?" --persona engineering --mode rerank
```
*Stage 1 fetches top-15 candidates via Hybrid RRF, and Stage 2 reranks them using FlashRank ONNX cross-attention to pass the top-5 cleanest chunks to Gemini.*

#### Query as a Standard Employee (Strict RBAC Enforced):
```bash
python src/document_retriever/rag_retriever.py "how much is performance bonus?" --persona employee --mode rerank
```
*Output: Pre-retrieval filters block confidential executive documents. FlashRank scores non-confidential docs near 0.0, and the LLM responds that it does not have enough information.*

#### Query as an Executive:
```bash
python src/document_retriever/rag_retriever.py "how much is performance bonus?" --persona executive --mode rerank
```
*Output: Retrieves confidential compensation tables, reranks highest-relevance excerpts to Rank #1, and generates exact figures with citations.*

---

## 🛡️ Security Best Practices
- **No Prompt-Based Security**: RBAC is enforced strictly at the database level (`where={"persona": {"$in": ...}}`), preventing prompt-injection attacks.
- **Zero Local Vector DB Commits**: `chroma_db/` and credential files are gitignored.
- **Deterministic Citations**: Answers strictly reference source documents and page numbers.

