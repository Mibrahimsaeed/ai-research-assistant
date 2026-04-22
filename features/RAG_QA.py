import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rag_engine.llm_client import ask_llm  # ✅ NEW (central LLM handler)

load_dotenv()

# Files
FAISS_INDEX_FILE = "data/faiss_index.index"
METADATA_FILE = "data/faiss_metadata.pkl"

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + metadata
index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


# -----------------------------
# RETRIEVE CONTEXT
# -----------------------------
def retrieve(query, k=5):
    query_vec = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


# -----------------------------
# BUILD PROMPT
# -----------------------------
def build_prompt(query, contexts):
    context_text = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
You are an AI research assistant.

Use the context below to answer the question.

If the context does not contain enough information, say you don't know.

Context:
{context_text}

Question:
{query}

Answer clearly and concisely.
"""

    return prompt


# -----------------------------
# GENERATE ANSWER (UPDATED)
# -----------------------------
def generate_answer(prompt):
    return ask_llm(prompt, temperature=0.3)


# -----------------------------
# MAIN CHAT LOOP
# -----------------------------
def chat():
    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        contexts = retrieve(query)
        prompt = build_prompt(query, contexts)
        answer = generate_answer(prompt)

        print("\n🧠 Answer:\n")
        print(answer)


if __name__ == "__main__":
    chat()