from app.embeddings import model
from app.chroma_store import search_chunks_mmr

query = "What are the two pre-training objectives used by BERT?"

results = search_chunks_mmr(
    query=query,
    model=model,
    top_k=5,
    fetch_k=20,
    lambda_param=0.7
)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["metadata"]["source"])
    print("Page:", result["metadata"]["page_number"])
    print("Similarity:", round(result["score"], 4))
    print("Chunk:", result["chunk"][:200])