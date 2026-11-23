import hashlib
from pathlib import Path

from document_loader import extract_pages
from chunking import chunk_document
from embeddings import generate_embeddings
from chroma_store import collection, store_chunks


def generate_document_id(pdf_path):

    file_content = Path(pdf_path).read_bytes()

    return hashlib.sha256(file_content).hexdigest()[:16]


def ingest_document(pdf_path):

    document_id = generate_document_id(pdf_path)

    source = Path(pdf_path).name

    # Check whether this document is already indexed
    existing = collection.get(
        where={"document_id": document_id},
        limit=1
    )

    if existing["ids"]:

        print(f"Document already indexed: {source}")
        print("Skipping ingestion.")

        return document_id

    # Step 1: Extract page-wise text
    pages = extract_pages(pdf_path)

    print("Pages extracted:", len(pages))

    # Step 2: Generate chunks with page metadata
    chunks, page_metadata = chunk_document(pages)

    print("Total chunks:", len(chunks))

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 4: Store in ChromaDB
    store_chunks(
        chunks,
        embeddings,
        page_metadata,
        document_id,
        source
    )

    print(f"Successfully indexed: {source}")

    return document_id