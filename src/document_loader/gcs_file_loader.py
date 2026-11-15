from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_community import GCSDirectoryLoader

# Mapping of file patterns to their required user persona / access level
DOCUMENT_PERSONA_MAP = {
    # All Employees / General Company-Wide
    "nexus_employee_handbook": "employee",
    "code_of_conduct_and_ethics": "employee",
    "office_safety_and_emergency": "employee",
    "benefits_and_insurance_guide": "employee",
    "parental_leave_policy": "employee",
    "travel_and_expense_policy": "employee",
    "ai_and_llm_usage_policy": "employee",

    # Human Resources
    "performance_review_policy": "hr",

    # Engineering & Architecture
    "cloud_architecture_standards": "engineering",
    "sdlc_and_release_process": "engineering",
    "project_aegis_functional_spec": "engineering",
    "project_titan_migration_plan": "engineering",

    # Security & Compliance
    "incident_response_playbook": "security",
    "information_security_policy": "security",
    "data_privacy_and_gdpr": "security",

    # Sales & Client Operations
    "client_onboarding_playbook": "sales",
    "project_orion_client_sow": "sales",

    # Finance & Procurement
    "procurement_and_vendor_policy": "finance",

    # Executive & Leadership (Confidential)
    "compensation_and_bonus": "executive",
    "product_roadmap": "executive",
}


def get_document_persona(file_path: str) -> str:
    """
    Infers the access persona for a document based on its filename.
    Defaults to 'employee' if no specific restriction matches.
    """
    clean_name = Path(file_path).stem.lower()
    for pattern, persona in DOCUMENT_PERSONA_MAP.items():
        if pattern in clean_name:
            return persona
    return "employee"


def gcs_file_loader() -> List[Document]:
    """
    Loads all PDFs from Google Cloud Storage and tags each page
    with its corresponding persona access level in metadata.
    """
    loader = GCSDirectoryLoader(
        project_name="poc-data-analytics-501104",
        bucket="amby-open-ragbench-bucket",
        prefix="nexus_long_pdfs/",
        loader_func=PyPDFLoader,
    )

    docs = loader.load()
    print(f"Successfully loaded {len(docs)} pages from GCS.")

    # Tag each document with persona metadata
    persona_counts = {}
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        filename = Path(source_path).name
        persona = get_document_persona(filename)
        doc.metadata["persona"] = persona
        doc.metadata["filename"] = filename
        persona_counts[persona] = persona_counts.get(persona, 0) + 1

    print("Document chunks tagged by persona:")
    for role, count in sorted(persona_counts.items()):
        print(f"  - {role.upper():12}: {count} pages")

    return docs


if __name__ == "__main__":
    gcs_file_loader()
