import chromadb

# Create a persistent local database
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or reuse a collection
collection = client.get_or_create_collection(
    name="research_documents",
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