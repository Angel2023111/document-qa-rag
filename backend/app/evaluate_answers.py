
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from embeddings import model
from chroma_store import search_chunks
from llm import generate_answer, evaluate_answer_correctness
from answer_evaluation_dataset import answer_evaluation_questions
from citation_evaluation import (
    evaluate_citations,
    evaluate_citation_relevance
)

def evaluate_answers(top_k=5):

    for index, item in enumerate(answer_evaluation_questions, start=1):

        question = item["question"]

        print("\n" + "=" * 60)
        print(f"Question {index}: {question}")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = search_chunks(
            question,
            model,
            top_k=top_k
        )

        # Step 2: Generate answer using the LLM
        generated_answer = generate_answer(
            question,
            retrieved_chunks
        )

        evaluation = evaluate_answer_correctness(
            question,
            item["expected_answer"],
            generated_answer
        )

        citation_result = evaluate_citations(
            generated_answer,
            retrieved_chunks
        )

        relevance_results = evaluate_citation_relevance(
            generated_answer,
            retrieved_chunks
        )

        print("\nCitation Evaluation:")
        print(f"Total Citations: {citation_result['total_citations']}")
        print(f"Valid Citations: {citation_result['valid_citations']}")
        print(
            f"Citation Precision: "
            f"{citation_result['citation_precision'] * 100:.2f}%"
        )

        print("\nCitation Relevance Evaluation:")
        print(
            f"Total Citations: "
            f"{relevance_results['total_citations']}"
        )

        print(
            f"Supported Citations: "
            f"{relevance_results['supported_citations']}"
        )

        print(
            f"Citation Relevance: "
            f"{relevance_results['citation_relevance'] * 100:.2f}%"
        )

        for citation in relevance_results["details"]:
            print(
                f"{citation['source']} - Page {citation['page']}: "
                f"{citation['supported']}"
            )
            print(f"Reason: {citation['reason']}")

        print("\nCorrectness Evaluation:")
        print(f"Score: {evaluation['score']}/5")
        print(f"Reason: {evaluation['reason']}")

        # Step 3: Display results
        print("\nExpected Answer:")
        print(item["expected_answer"])

        print("\nGenerated Answer:")
        print(generated_answer)

        print("\nExpected Citation:")
        print(f"Source: {item['expected_source']}")
        print(f"Pages: {item['expected_pages']}")

        print("\nRetrieved Sources:")

        for result in retrieved_chunks:
            metadata = result["metadata"]

            print(
                f"Source: {metadata['source']}, "
                f"Page: {metadata['page_number']}, "
                f"Similarity: {result['score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_answers(top_k=10)
