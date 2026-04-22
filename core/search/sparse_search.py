import json
import os
from rank_bm25 import BM25Okapi

# -----------------------------
# CONFIG
# -----------------------------
CHUNKS_FILE = os.path.join("data", "chunks.json")


# -----------------------------
# BM25 RETRIEVER CLASS
# -----------------------------
class BM25Retriever:
    def __init__(self):
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

        self._load_data()
        self._build_index()

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def _load_data(self):
        if not os.path.exists(CHUNKS_FILE):
            raise Exception(f"Chunks file not found at {CHUNKS_FILE}")

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # -----------------------------
    # TOKENIZATION (simple baseline)
    # -----------------------------
    def _tokenize(self, text):
        return text.lower().split()

    # -----------------------------
    # BUILD BM25 INDEX
    # -----------------------------
    def _build_index(self):
        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # -----------------------------
    # SEARCH FUNCTION
    # -----------------------------
    def search(self, query, top_k=10):
        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "paper_id": self.chunks[idx].get("paper_id", None),
                "score": float(scores[idx])
            })

        return results


# -----------------------------
# GLOBAL INSTANCE (IMPORTANT)
# -----------------------------
bm25_retriever = BM25Retriever()


# -----------------------------
# FUNCTION API (FOR ROUTERS)
# -----------------------------
def bm25_search(query, top_k=10):
    """
    Clean interface used by routers and hybrid search
    """
    return bm25_retriever.search(query, top_k=top_k)


# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    query = input("Enter query: ")

    results = bm25_search(query, top_k=5)

    print("\n🔍 BM25 Results:\n")

    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(r["text"][:200])
        print("-" * 50)