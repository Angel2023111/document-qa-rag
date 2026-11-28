
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from embeddings import model
from chroma_store import search_chunks, search_chunks_mmr
from evaluation_dataset import evaluation_questions


def evaluate_retrieval(top_k=5, method="similarity", lambda_param=0.7):

    total_questions = len(evaluation_questions)

    total_hits = 0
    total_relevant = 0
    total_retrieved_relevant = 0
    total_reciprocal_rank = 0

    for item in evaluation_questions:

        query = item["question"]
        expected_source = item["expected_source"]
        expected_pages = set(item["expected_pages"])

        # Select retrieval strategy
        if method == "similarity":
            retrieved_chunks = search_chunks(
                query,
                model,
                top_k=top_k
            )

        elif method == "mmr":
            retrieved_chunks = search_chunks_mmr(
                query,
                model,
                top_k=top_k,
                fetch_k=40,
                lambda_param=lambda_param
            )

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        retrieved_pages = set()
        reciprocal_rank = 0

        for rank, result in enumerate(retrieved_chunks, start=1):

            metadata = result["metadata"]

            if metadata["source"] == expected_source:

                page_number = metadata["page_number"]
                retrieved_pages.add(page_number)

                # Find the first relevant result
                if page_number in expected_pages and reciprocal_rank == 0:
                    reciprocal_rank = 1 / rank

        total_reciprocal_rank += reciprocal_rank

        # Count relevant pages found
        relevant_found = len(
            retrieved_pages.intersection(expected_pages)
        )

        total_relevant += len(expected_pages)
        total_retrieved_relevant += relevant_found

        # Hit = at least one expected page retrieved
        if relevant_found > 0:
            total_hits += 1

    hit_rate = total_hits / total_questions

    recall = (
        total_retrieved_relevant / total_relevant
        if total_relevant > 0 else 0
    )

    mrr = total_reciprocal_rank / total_questions

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "mrr": mrr
    }


if __name__ == "__main__":

    # Make sure we are evaluating the selected baseline collection
    print("========== RETRIEVAL EVALUATION ==========")
    print("Collection: research_documents_500_50")
    print(f"Questions: {len(evaluation_questions)}")

    configurations = [
        ("Similarity", "similarity", None),
        ("MMR (0.7)", "mmr", 0.7),
        ("MMR (0.5)", "mmr", 0.5),
        ("MMR (0.3)", "mmr", 0.3)
    ]

    results = []

    for top_k in [5, 10]:

        print(f"\n\n========== TOP K = {top_k} ==========")

        for name, method, lambda_param in configurations:

            metrics = evaluate_retrieval(
                top_k=top_k,
                method=method,
                lambda_param=lambda_param
            )

            results.append({
                "name": name,
                "top_k": top_k,
                **metrics
            })

            print(
                f"{name:<15} | "
                f"Hit Rate: {metrics['hit_rate']:.2%} | "
                f"Recall: {metrics['recall']:.2%} | "
                f"MRR: {metrics['mrr']:.4f}"
            )

    print("\n\n========== FINAL COMPARISON ==========")
    print(
        f"{'Method':<15} {'K':<5} "
        f"{'Hit Rate':<12} {'Recall':<12} {'MRR':<10}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['name']:<15} "
            f"{result['top_k']:<5} "
            f"{result['hit_rate']:<12.2%} "
            f"{result['recall']:<12.2%} "
            f"{result['mrr']:<10.4f}"
        )
