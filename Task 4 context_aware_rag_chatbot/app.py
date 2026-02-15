# app.py
import streamlit as st
from rag.rag_chain import get_rag_chain
from feedback_logger import save_feedback

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("### 🔍 System Info")
    st.markdown("""
    - **LLM:** LLaMA 3.1 (Groq)
    - **Vector DB:** Qdrant
    - **Embeddings:** MiniLM
    - **Retriever Top-K:** 4
    """)

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🧠 Context-Aware RAG Chatbot")
st.caption("Ask questions about AI, ML, NLP, and RAG concepts.")

# --------------------------------------------------
# Initialize RAG Chain
# --------------------------------------------------
if "rag_chain" not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state.rag_chain = get_rag_chain()

rag_chain = st.session_state.rag_chain

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session"

session_id = st.session_state.session_id

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for i, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Feedback only for assistant messages
        if message["role"] == "assistant":

            col1, col2, col3 = st.columns([1,1,8])

            with col1:
                if st.button("👍", key=f"up_{i}"):
                    st.session_state.feedback[i] = "up"
                    save_feedback(message["content"], "up")
                    st.toast("Feedback saved 👍")

            with col2:
                if st.button("👎", key=f"down_{i}"):
                    st.session_state.feedback[i] = "down"
                    save_feedback(message["content"], "down")
                    st.toast("Feedback saved 👎")

            selected = st.session_state.feedback.get(i)
            if selected == "up":
                st.caption("You marked this response as helpful 👍")
            elif selected == "down":
                st.caption("You marked this response as not helpful 👎")

# --------------------------------------------------
# User Input
# --------------------------------------------------
if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Thinking..."):
                result = rag_chain(
                    prompt,
                    config={"configurable": {"session_id": session_id}}
                )

            full_response = str(result["answer"])

        except Exception as e:
            full_response = "⚠️ Error generating response."
            st.error(str(e))

        st.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        msg_index = len(st.session_state.messages) - 1


        col1, col2, col3 = st.columns([1,1,8])

        with col1:
            if st.button("👍", key=f"up_{msg_index}"):
                st.session_state.feedback[msg_index] = "up"
                save_feedback(full_response, "up")
                st.toast("Feedback saved 👍")

        with col2:
            if st.button("👎", key=f"down_{msg_index}"):
                st.session_state.feedback[msg_index] = "down"
                save_feedback(full_response, "down")
                st.toast("Feedback saved 👎")
