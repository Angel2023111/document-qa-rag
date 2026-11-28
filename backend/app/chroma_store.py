import os
import chromadb
import numpy as np

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Each experiment gets its own collection
collection_name = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "research_documents"
)

collection = client.get_or_create_collection(
    name=collection_name,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


def store_chunks(
    chunks,
    embeddings,
    page_metadata,
    document_id,
    source
):

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = []

    for i, page_info in enumerate(page_metadata):

        metadatas.append({
            "source": source,
            "document_id": document_id,
            "chunk_index": i,
            "page_number": page_info["page_number"]
        })

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("Stored chunks:", collection.count())


def search_chunks(query, model, top_k=3):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    for i in range(len(results["ids"][0])):

        retrieved.append({
            "chunk": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]
        })

    return retrieved


def search_chunks_mmr(query, model, top_k=5, fetch_k=20, lambda_param=0.7):

    # Step 1: Generate query embedding
    query_embedding = model.encode(query)

    # Step 2: Retrieve more candidates than we actually need
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=fetch_k,
        include=["documents", "metadatas", "embeddings", "distances"]
    )

    candidates = []

    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "chunk": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "embedding": np.array(results["embeddings"][0][i]),
            "score": 1 - results["distances"][0][i]
        })

    # Step 3: Normalize embeddings
    def normalize(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    query_embedding = normalize(query_embedding)

    for candidate in candidates:
        candidate["embedding"] = normalize(candidate["embedding"])

    # Step 4: Apply MMR
    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:

        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:

            relevance = np.dot(
                candidate["embedding"],
                query_embedding
            )

            if selected:
                redundancy = max(
                    np.dot(candidate["embedding"], item["embedding"])
                    for item in selected
                )
            else:
                redundancy = 0

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        remaining = [
            candidate for candidate in remaining
            if candidate["id"] != best_candidate["id"]
        ]

    # Step 5: Return selected chunks in MMR order
    retrieved = []

    for candidate in selected:
        retrieved.append({
            "chunk": candidate["chunk"],
            "metadata": candidate["metadata"],
            "score": candidate["score"]
        })

    return retrieved
