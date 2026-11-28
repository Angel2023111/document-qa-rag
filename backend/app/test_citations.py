from citation_evaluation import evaluate_citations

generated_answer = """
BERT uses bidirectional representations.
[Source: bert.pdf, Page: 1]

BERT uses two pre-training objectives.
[Source: bert.pdf, Page: 4]

The Transformer uses positional encoding.
[Source: sample.pdf, Page: 6]
"""

retrieved_chunks = [
    {
        "metadata": {
            "source": "bert.pdf",
            "page_number": 1
        }
    },
    {
        "metadata": {
            "source": "bert.pdf",
            "page_number": 4
        }
    }
]

result = evaluate_citations(generated_answer, retrieved_chunks)

print(result)