import pymupdf

from chunking import chunk_text
from embeddings import model, generate_embeddings
from chroma_store import store_chunks, search_chunks
from llm import generate_answer


# 1. Extract PDF text
pdf_path = "documents/sample.pdf"

document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()


# 2. Chunk the document
chunks = chunk_text(full_text)

print("Total chunks:", len(chunks))


# 3. Generate embeddings
embeddings = generate_embeddings(chunks)


# 4. Store in ChromaDB
store_chunks(chunks, embeddings)


# 5. Ask a question
query = input("\nAsk a question about your document: ")


# 6. Retrieve relevant chunks
retrieved_chunks = search_chunks(
    query,
    model,
    top_k=3
)


# 7. Generate answer using LLM
answer = generate_answer(
    query,
    retrieved_chunks
)


# 8. Display answer
print("\n" + "=" * 60)
print("QUESTION:", query)
print("=" * 60)

print("\nANSWER:")
print(answer)

print("\n" + "=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for i, result in enumerate(retrieved_chunks):

    print(f"\nSource {i+1}")
    print("Metadata:", result["metadata"])
    print("Similarity:", round(result["score"], 4))