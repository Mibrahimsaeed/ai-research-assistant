import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

INPUT_FILE = "data/chunks.json"
FAISS_INDEX_FILE = "data/faiss_index.index"
METADATA_FILE = "data/faiss_metadata.pkl"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings():
    with open(INPUT_FILE, "r") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print("🔄 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata (important!)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Embeddings + FAISS index created!")


if __name__ == "__main__":
    create_embeddings()