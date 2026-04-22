from routers import route_query
from hybrid_search import hybrid_search
from context_builder import build_context
from core.rag_engine.llm_client import generate_answer

def process(query):
    routed = route_query(query)

    # 2. hybrid search (does dense + sparse + fusion + rerank)
    results = hybrid_search(query, k=routed["k"])

    # 3. build context
    context = build_context(results)
    answer=generate_answer(query, context)

    return {
        "answer": answer,
        "context": context,
        "intent": routed["intent"],
        "results": results
    }


if __name__ == "__main__":
    query = input("Enter query: ")

    result = process(query)

    print("\n==============================")
    print("INTENT:", result["intent"])

    print("\nANSWER:\n")
    print(result["answer"])