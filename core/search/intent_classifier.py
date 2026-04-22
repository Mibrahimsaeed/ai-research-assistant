from sentence_transformers import SentenceTransformer, util
from intent_examples import INTENT_EXAMPLES

model = SentenceTransformer("all-MiniLM-L6-v2")


# Precompute intent embeddings
intent_embeddings = {}

for intent, examples in INTENT_EXAMPLES.items():
    intent_embeddings[intent] = model.encode(examples, normalize_embeddings=True)


def classify_intent(query: str):
    query_emb = model.encode(query, normalize_embeddings=True)

    best_intent = None
    best_score = -1

    for intent, emb_list in intent_embeddings.items():
        scores = util.cos_sim(query_emb, emb_list)[0]
        max_score = float(scores.max())

        if max_score > best_score:
            best_score = max_score
            best_intent = intent

    return {
        "intent": best_intent,
        "confidence": best_score
    }




if __name__ == "__main__":
    while True:
        q = input("Query: ")
        result = classify_intent(q)

        print("\nIntent:", result["intent"])
        print("Confidence:", round(result["confidence"], 3))
        print("-" * 40)