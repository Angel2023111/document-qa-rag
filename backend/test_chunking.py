from app.chunking import chunk_document

pages = [
    {
        "page_number": 1,
        "text": """
        BERT is a bidirectional Transformer encoder.
        It uses masked language modeling.
        This helps it understand contextual relationships.
        """
    }
]

chunks, metadata = chunk_document(pages, chunk_size=100, overlap=30)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)
    print("Metadata:", metadata[i])