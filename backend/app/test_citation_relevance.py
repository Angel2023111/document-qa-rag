from chroma_store import search_chunks
from embeddings import model
from llm import generate_answer
from citation_evaluation import evaluate_citation_relevance

question = "What is the main objective of BERT?"

retrieved_chunks = search_chunks(
    question,
    model,
    top_k=10
)

generated_answer = generate_answer(
    question,
    retrieved_chunks
)

result = evaluate_citation_relevance(
    generated_answer,
    retrieved_chunks
)

print("\nGenerated Answer:")
print(generated_answer)

print("\nCitation Relevance:")
print(f"Total: {result['total_citations']}")
print(f"Supported: {result['supported_citations']}")
print(
    f"Relevance Precision: "
    f"{result['citation_relevance'] * 100:.2f}%"
)

for item in result["details"]:
    print(f"\n{item['source']} - Page {item['page']}")
    print(f"Supported: {item['supported']}")
    print(f"Reason: {item['reason']}")