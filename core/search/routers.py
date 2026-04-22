from intent_classifier import classify_intent
from routing_policy import ROUTING_POLICY
from hybrid_search import hybrid_search
from sparse_search import bm25_search
from dense_search import dense_search

def route_query(query):

    intent_result = classify_intent(query)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]

    policy = ROUTING_POLICY[intent]

    k = policy["k"]

    # Step 1: retrieve more candidates than needed
    dense_results = dense_search(query, k=20)
    sparse_results = bm25_search(query, top_k=20)

    return {
        "intent": intent,
        "confidence": confidence,
        "policy": policy,
        "dense": dense_results,
        "sparse": sparse_results,
        "k": k
    }