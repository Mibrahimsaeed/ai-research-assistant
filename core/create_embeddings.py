import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

from state_manager import set_status, STATUS, should_process

INPUT_FILE = "data/chunks.json"
FAISS_INDEX_FILE = "data/faiss_index.index"
METADATA_FILE = "data/faiss_metadata.pkl"
PAPERS_METADATA_FILE = "data/metadata.json"

# Load model (only once)
model = SentenceTransformer("BAAI/bge-large-en")


def create_embeddings():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    with open(INPUT_FILE, "r") as f:
        chunks = json.load(f)

    with open(PAPERS_METADATA_FILE, "r") as f:
        papers_metadata = json.load(f)

    # -----------------------------
    # FILTER: ONLY UNPROCESSED
    # -----------------------------
    filtered_chunks = [
        chunk for chunk in chunks
        if should_process(papers_metadata, chunk["paper_id"])
    ]

    if not filtered_chunks:
        print("✅ No new chunks to process.")
        return

    texts = [chunk["text"] for chunk in filtered_chunks]

    print(f"🔄 Creating embeddings for {len(texts)} chunks...")

    # -----------------------------
    # GENERATE EMBEDDINGS
    # -----------------------------
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    # -----------------------------
    # LOAD OR CREATE FAISS INDEX
    # -----------------------------
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        print("📂 Loaded existing FAISS index")
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        print("🆕 Created new FAISS index")

    # -----------------------------
    # ADD NEW EMBEDDINGS
    # -----------------------------
    index.add(embeddings)

    # Save index
    faiss.write_index(index, FAISS_INDEX_FILE)

    # -----------------------------
    # LOAD OR APPEND METADATA
    # -----------------------------
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            existing_meta = pickle.load(f)
    else:
        existing_meta = []

    existing_meta.extend(filtered_chunks)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(existing_meta, f)

    # -----------------------------
    # UPDATE STATUS (ONLY PROCESSED)
    # -----------------------------
    processed_papers = set(chunk["paper_id"] for chunk in filtered_chunks)

    for paper_id in processed_papers:
        papers_metadata = set_status(
            papers_metadata,
            paper_id,
            STATUS["EMBEDDED"]
        )

    # Save updated metadata
    with open(PAPERS_METADATA_FILE, "w") as f:
        json.dump(papers_metadata, f, indent=4)

    print("✅ Embeddings created and stored successfully!")


if __name__ == "__main__":
    create_embeddings()