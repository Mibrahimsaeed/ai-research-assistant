import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from core.config import FAISS_INDEX_FILE, FAISS_METADATA_FILE


# -----------------------------
# MODEL (must match embedding step)
# -----------------------------
model = SentenceTransformer("BAAI/bge-large-en")


# -----------------------------
# LOAD FAISS + METADATA (ONCE)
# -----------------------------
index = faiss.read_index(FAISS_INDEX_FILE)

with open(FAISS_METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


# -----------------------------
# CORE SEARCH FUNCTION
# -----------------------------
def dense_search(query, k=5):
    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []

    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "text": metadata[idx]["text"],
                "paper_id": metadata[idx].get("paper_id"),
                "score": float(distances[0][i])
            })

    return results


# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    q = input("Enter query: ")

    results = dense_search(q)

    print("\n🔍 Dense Search Results:\n")

    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(r["text"][:300])
        print("-" * 50)