import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_FILE = "data/faiss_index.index"
METADATA_FILE = "data/faiss_metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index + metadata
index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


def search(query, k=5):
    query_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search(query)

    print("\n🔍 Top Results:\n")
    for r in results:
        print(f"📄 {r['title']}")
        print(r["text"][:300])
        print("-" * 50)