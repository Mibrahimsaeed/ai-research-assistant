from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, candidates, top_k=5):
    """
    query: string
    candidates: list of dicts with 'text'
    """

    pairs = [(query, item["text"]) for item in candidates]

    scores = reranker.predict(pairs)

    # Attach scores
    for i, item in enumerate(candidates):
        item["rerank_score"] = float(scores[i])

    # Sort by rerank score
    ranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]