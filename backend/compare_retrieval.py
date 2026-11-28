from app.embeddings import model
from app.chroma_store import search_chunks, search_chunks_mmr

query = "What are the two pre-training objectives used by BERT?"

print("\n========== SIMILARITY SEARCH ==========")
normal_results = search_chunks(query, model, top_k=5)

for i, result in enumerate(normal_results, start=1):
    print(f"\nResult {i}")
    print("Page:", result["metadata"]["page_number"])
    print("Similarity:", round(result["score"], 4))
    print("Chunk:", result["chunk"][:400])

print("\n========== MMR SEARCH ==========")
mmr_results = search_chunks_mmr(
    query, model, top_k=5, fetch_k=20, lambda_param=0.7
)

for i, result in enumerate(mmr_results, start=1):
    print(f"\nResult {i}")
    print("Page:", result["metadata"]["page_number"])
    print("Similarity:", round(result["score"], 4))
    print("Chunk:", result["chunk"][:400])