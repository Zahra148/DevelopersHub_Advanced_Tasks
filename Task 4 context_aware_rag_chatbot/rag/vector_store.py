# rag/vector_store.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "RAG_Collection")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_URL or QDRANT_API_KEY missing in .env")

# --------------------------------------------------
# Embeddings (Cached)
# --------------------------------------------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# Qdrant Client
# --------------------------------------------------
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

# --------------------------------------------------
# Vector Store
# --------------------------------------------------
@st.cache_resource
def get_vector_store():
    embeddings = get_embedding_model()
    client = get_qdrant_client()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
