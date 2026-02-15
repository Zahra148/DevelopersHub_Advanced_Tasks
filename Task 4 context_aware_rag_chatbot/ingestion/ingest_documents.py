#/ingestion/ingest_documents.py
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.vector_store import get_vector_store


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# Absolute path to data folder
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"


# --------------------------------------------------
# Load Documents
# --------------------------------------------------
def load_documents() -> list[Document]:
    """
    Load all .txt files from data directory
    """
    documents = []

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Data folder not found at: {DATA_PATH}")

    for file_path in DATA_PATH.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    print(f"✅ Loaded {len(documents)} raw documents")
    return documents


# --------------------------------------------------
# Split Documents into Chunks
# --------------------------------------------------
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# --------------------------------------------------
# Ingest into Qdrant
# --------------------------------------------------
def ingest():
    """
    Full ingestion pipeline:
    Load → Split → Embed → Store
    """
    print("🚀 Starting ingestion process...")

    documents = load_documents()
    chunks = split_documents(documents)

    if not chunks:
        print("⚠️ No chunks to ingest.")
        return

    vector_store = get_vector_store()

    # Optional: Clear collection before re-ingesting
    # Uncomment below if you want fresh ingestion each time
    # vector_store.delete_collection()

    vector_store.add_documents(chunks)

    print("🎉 Documents successfully ingested into Qdrant!")


# --------------------------------------------------
# Run Ingestion
# --------------------------------------------------
if __name__ == "__main__":
    ingest()
