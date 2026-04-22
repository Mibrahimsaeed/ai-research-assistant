from dense_search import dense_search
from sparse_search import bm25_search
from reranker import rerank

RRF_K = 60


# -----------------------------
# RECIPROCAL RANK FUSION
# -----------------------------
def reciprocal_rank_fusion(dense_results, sparse_results):
    scores = {}

    for rank, item in enumerate(dense_results):
        key = item["text"]

        if key not in scores:
            scores[key] = {"data": item, "score": 0}

        scores[key]["score"] += 1 / (RRF_K + rank)

    for rank, item in enumerate(sparse_results):
        key = item["text"]

        if key not in scores:
            scores[key] = {"data": item, "score": 0}

        scores[key]["score"] += 1 / (RRF_K + rank)

    fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    return [x["data"] for x in fused]


# -----------------------------
# HYBRID SEARCH PIPELINE
# -----------------------------
def hybrid_search(query, k=5):
    dense_results = dense_search(query, k=20)
    sparse_results = bm25_search(query, top_k=20)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    final_results = rerank(query, fused, top_k=k)

    return final_results


# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    q = input("Enter query: ")

    results = hybrid_search(q)

    print("\n🔍 Hybrid Results:\n")

    for r in results:
        print(r["text"][:200])
        print("-" * 50)