from embeddings import model
from chroma_store import search_chunks
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation_dataset import evaluation_questions


def evaluate_retrieval(top_k=5):

    total_questions = len(evaluation_questions)

    total_hits = 0
    total_relevant = 0
    total_retrieved_relevant = 0

    for item in evaluation_questions:

        query = item["question"]
        expected_source = item["expected_source"]
        expected_pages = set(item["expected_pages"])

        retrieved_chunks = search_chunks(
            query,
            model,
            top_k=top_k
        )

        if "two pre-training objectives" in query:
            print("\n--- DEBUG: Retrieved chunks ---")

            for rank, result in enumerate(retrieved_chunks, start=1):
                metadata = result["metadata"]

                print(f"\nRank: {rank}")
                print(f"Source: {metadata['source']}")
                print(f"Page: {metadata['page_number']}")
                print(f"Similarity: {result['score']:.4f}")
                print(f"Chunk: {result['chunk'][:300]}")

        retrieved_pages = set()

        for result in retrieved_chunks:

            metadata = result["metadata"]

            if metadata["source"] == expected_source:
                retrieved_pages.add(metadata["page_number"])

        # Count relevant pages found
        relevant_found = len(
            retrieved_pages.intersection(expected_pages)
        )

        total_relevant += len(expected_pages)
        total_retrieved_relevant += relevant_found

        # Hit = at least one expected page retrieved
        if relevant_found > 0:
            total_hits += 1

        print(f"\nQuestion: {query}")
        print(f"Expected source: {expected_source}")
        print(f"Expected pages: {sorted(expected_pages)}")
        print(f"Retrieved relevant pages: {sorted(retrieved_pages.intersection(expected_pages))}")
        print(f"Relevant pages found: {relevant_found}")

    hit_rate = total_hits / total_questions

    recall = (
        total_retrieved_relevant / total_relevant
        if total_relevant > 0 else 0
    )

    print("\n========== EVALUATION RESULTS ==========")

    print(f"Hit Rate@{top_k}: {hit_rate:.2%}")
    print(f"Recall@{top_k}: {recall:.2%}")


if __name__ == "__main__":
    evaluate_retrieval(top_k=5)