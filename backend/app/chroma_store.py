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


def store_chunks(chunks, embeddings):

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": "sample.pdf",
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

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