#/utils/test_env.py
from dotenv import load_dotenv
import os

load_dotenv()

print("QDRANT_URL:", os.getenv("QDRANT_URL"))
print("QDRANT_API_KEY:", "Loaded" if os.getenv("QDRANT_API_KEY") else "Missing")
print("GROQ_API_KEY:", "Loaded" if os.getenv("GROQ_API_KEY") else "Missing")
