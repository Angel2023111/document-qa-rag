import pymupdf

from chunking import chunk_text
from embeddings import model, generate_embeddings
from chroma_store import store_chunks, search_chunks


# Extract PDF text
pdf_path = "documents/sample.pdf"

document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()


# Chunking
chunks = chunk_text(full_text)

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Store in ChromaDB
store_chunks(chunks, embeddings)


# Search
queries = [
    "What is the main idea behind the Transformer?",
    "Why does the Transformer use attention mechanisms?",
    "What are the advantages of the Transformer over RNNs?"
]

for query in queries:

    print("\n" + "=" * 60)
    print("QUESTION:", query)
    print("=" * 60)

    results = search_chunks(query, model, top_k=3)

    for i, result in enumerate(results):

        print(f"\nResult {i+1}")
        print("Similarity:", round(result["score"], 4))
        print("Metadata:", result["metadata"])
        print("Text:", result["chunk"])