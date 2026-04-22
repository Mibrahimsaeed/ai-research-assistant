ROUTING_POLICY = {
    "FACTUAL": {
        "k": 5,
        "dense_weight": 0.7,
        "bm25_weight": 0.3
    },

    "CONCEPTUAL": {
        "k": 8,
        "dense_weight": 0.8,
        "bm25_weight": 0.2
    },

    "COMPARATIVE": {
        "k": 10,
        "dense_weight": 0.5,
        "bm25_weight": 0.5
    },

    "EXPLORATORY": {
        "k": 15,
        "dense_weight": 0.4,
        "bm25_weight": 0.6
    }
}