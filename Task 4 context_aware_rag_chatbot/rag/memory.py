from langchain_core.chat_history import InMemoryChatMessageHistory


def get_memory():
    """
    Returns in-memory chat history object (LangChain v1 compatible)
    """
    return InMemoryChatMessageHistory()
