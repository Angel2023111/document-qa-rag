from app.embeddings import model
from app.chroma_store import search_chunks_mmr

query = "What are the two pre-training objectives used by BERT?"

for lambda_value in [1.0, 0.7, 0.5, 0.3]:

    print(f"\n========== LAMBDA = {lambda_value} ==========")

    results = search_chunks_mmr(
        query=query,
        model=model,
        top_k=5,
        fetch_k=20,
        lambda_param=lambda_value
    )

    for i, result in enumerate(results, start=1):
        print(
            f"Result {i}: "
            f"Page {result['metadata']['page_number']} | "
            f"Similarity {result['score']:.4f}"
        )