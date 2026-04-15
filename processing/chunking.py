import json
import os

INPUT_FILE = "data/clean_papers.json"
OUTPUT_FILE = "data/chunks.json"

CHUNK_SIZE = 400
OVERLAP = 100


def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


def process_chunks():
    with open(INPUT_FILE, "r") as f:
        papers = json.load(f)

    all_chunks = []

    for paper in papers:
        title = paper["title"]
        text = paper["clean_text"]

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "title": title,
                "chunk_id": idx,
                "text": chunk
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("✅ Chunking complete!")


if __name__ == "__main__":
    process_chunks()