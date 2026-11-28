import json
import re

from llm import client, MODEL_NAME

def evaluate_citations(generated_answer, retrieved_chunks):

    valid_sources = set()

    for result in retrieved_chunks:
        metadata = result["metadata"]

        valid_sources.add(
            (metadata["source"], metadata["page_number"])
        )

    # Extract citations from the generated answer
    import re

    pattern = r"[\[【]?Source:\s*([^,\]】]+),\s*Page:\s*(\d+)[\]】]?"

    citations = re.findall(pattern, generated_answer)

    if not citations:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "citation_precision": 0
        }

    valid_count = 0

    for source, page in citations:
        citation = (source.strip(), int(page))

        if citation in valid_sources:
            valid_count += 1

    precision = valid_count / len(citations)

    return {
        "total_citations": len(citations),
        "valid_citations": valid_count,
        "citation_precision": precision
    }


def evaluate_citation_relevance(generated_answer, retrieved_chunks):

    pattern = r"[\[【]?Source:\s*([^,\]】]+),\s*Page:\s*(\d+)[\]】]?"

    citations = re.findall(pattern, generated_answer)

    results = []

    for source, page in citations:
        source = source.strip()
        page = int(page)

        # Collect all retrieved text from this source and page
        page_chunks = [
            result["chunk"]
            for result in retrieved_chunks
            if result["metadata"]["source"] == source
            and result["metadata"]["page_number"] == page
        ]

        if not page_chunks:
            results.append({
                "source": source,
                "page": page,
                "supported": False,
                "reason": "No retrieved text available for this citation."
            })
            continue

        page_context = "\n".join(page_chunks)

        judge_prompt = f"""
        You are evaluating citations in a document question-answering system.

        Generated answer:
        {generated_answer}

        Cited source:
        {source}, page {page}

        Actual text retrieved from this cited page:
        {page_context}

        Determine whether the cited page provides evidence supporting
        at least one relevant factual claim in the generated answer.

        Rules:
        1. Use only the actual page text provided.
        2. Do not use outside knowledge.
        3. Mark supported as true only if the page provides relevant evidence.
        4. Do not mark a citation supported merely because it belongs
           to the correct document.
        5. A page may support part of an answer without supporting every claim.

        Return ONLY valid JSON:
        {{
            "supported": true,
            "reason": "Brief explanation."
        }}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0
        )

        result = json.loads(
            response.choices[0].message.content
        )

        results.append({
            "source": source,
            "page": page,
            "supported": result["supported"],
            "reason": result["reason"]
        })

    total = len(results)
    supported_count = sum(
        1 for result in results if result["supported"]
    )

    relevance_precision = (
        supported_count / total if total > 0 else 0
    )

    return {
        "total_citations": total,
        "supported_citations": supported_count,
        "citation_relevance": relevance_precision,
        "details": results
    }