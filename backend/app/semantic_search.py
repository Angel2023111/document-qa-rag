from sentence_transformers import SentenceTransformer, util

from embeddings import model
from sentence_transformers import util


def search(query, chunks, embeddings, top_k=3):

    # Convert user question into an embedding
    query_embedding = model.encode(query)

    # Calculate cosine similarity
    similarities = util.cos_sim(
        query_embedding,
        embeddings
    )[0]

    # Get indices of top K similar chunks
    top_results = similarities.argsort(
        descending=True
    )[:top_k]

    results = []

    for idx in top_results:
        results.append({
            "chunk": chunks[int(idx)],
            "score": float(similarities[idx])
        })

    return results