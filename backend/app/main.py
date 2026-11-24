from embeddings import model
from chroma_store import search_chunks
from llm import generate_answer


while True:

    query = input(
        "\nAsk a question (type exit to quit): "
    )

    if query.lower() == "exit":
        break

    if not query.strip():
        continue

    # Retrieve relevant chunks
    retrieved_chunks = search_chunks(
        query,
        model,
        top_k=5
    )

    if not retrieved_chunks:
        print("No relevant information found.")
        continue

    # Generate answer
    answer = generate_answer(
        query,
        retrieved_chunks
    )

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    for result in retrieved_chunks:

        metadata = result["metadata"]

        print(
            f"Document: {metadata['source']}, "
            f"Page: {metadata['page_number']}, "
            f"Similarity: {result['score']:.4f}"
        )