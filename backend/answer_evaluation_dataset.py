
answer_evaluation_questions = [
    {
        "question": "What is the main objective of BERT?",
        "expected_answer": (
            "BERT aims to pre-train deep bidirectional "
            "representations by jointly conditioning on "
            "both left and right context in all layers."
        ),
        "expected_source": "bert.pdf",
        "expected_pages": [1]
    },
    {
        "question": "What are the two pre-training objectives used by BERT?",
        "expected_answer": (
            "BERT uses Masked Language Modeling (MLM) "
            "and Next Sentence Prediction (NSP)."
        ),
        "expected_source": "bert.pdf",
        "expected_pages": [4]
    },
    {
        "question": "How does BERT use the Transformer architecture?",
        "expected_answer": (
            "BERT uses a multi-layer bidirectional "
            "Transformer encoder to learn contextual "
            "representations of text."
        ),
        "expected_source": "bert.pdf",
        "expected_pages": [3]
    },
    {
        "question": "Why does the Transformer use positional encoding?",
        "expected_answer": (
            "Positional encoding provides information "
            "about the position of tokens in a sequence, "
            "because the Transformer does not use recurrence "
            "or convolution to represent token order."
        ),
        "expected_source": "sample.pdf",
        "expected_pages": [6]
    },
    {
        "question": "What is multi-head attention?",
        "expected_answer": (
            "Multi-head attention allows the model to "
            "jointly attend to information from different "
            "representation subspaces at different positions."
        ),
        "expected_source": "sample.pdf",
        "expected_pages": [4, 5]
    }
]
