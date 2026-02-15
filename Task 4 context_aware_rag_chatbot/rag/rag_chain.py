# rag/rag_chain.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.vector_store import get_vector_store

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in .env")

# --------------------------------------------------
# Prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer ONLY from the provided context. "
        "If the answer is not in the context, say: "
        "'I don't know based on the provided documents.'"
    ),
    ("system", "Context:\n{context}"),
    ("human", "{question}")
])

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs)

# --------------------------------------------------
# Memory Store (Streamlit-safe)
# --------------------------------------------------
if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}

store = st.session_state.memory_store

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --------------------------------------------------
# Cached LLM
# --------------------------------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5
    )

# --------------------------------------------------
# Build RAG Chain (Stable Version)
# --------------------------------------------------
@st.cache_resource
def get_rag_chain():

    llm = get_llm()

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Base chain expects dict input
    base_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Final callable function
    def invoke(question: str, config=None):

        # Retrieve documents
        docs = retriever.invoke(question)

        # Prepare input dict for chain
        inputs = {
            "question": question,
            "context": format_docs(docs)
        }

        # Generate answer
        answer = chain_with_memory.invoke(
            inputs,
            config=config
        )

        return {
            "answer": answer
        }

    return invoke
