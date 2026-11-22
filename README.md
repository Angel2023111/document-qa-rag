# Document QA System using Retrieval-Augmented Generation (RAG)

A document question-answering system that retrieves relevant information from user-uploaded documents using semantic search and vector embeddings.

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- FastAPI (planned)
- React (planned)

## Current Features

- PDF text extraction using PyMuPDF.
- Fixed-size text chunking with configurable overlap.
- Text embeddings using `all-MiniLM-L6-v2`.
- Semantic search using cosine similarity.
- Top-K relevant chunk retrieval.
- Persistent vector storage using ChromaDB.
- Metadata storage for document chunks.

## Architecture

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Search

## Project Structure

```text
document-qa-rag/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── semantic_search.py
│   │   └── chroma_store.py
│   └── requirements.txt
├── evaluation/
└── README.md
```

## Setup Instructions

1. Clone the repository.

2. Navigate to the backend directory.

3. Create and activate a virtual environment.

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Place a PDF file inside `backend/documents/` named `sample.pdf`.

6. Run the application from the backend directory:

```bash
python app/main.py
```

## Roadmap

- [x] PDF text extraction
- [x] Fixed-size chunking
- [x] Embedding generation
- [x] Semantic search
- [x] ChromaDB integration and persistence
- [ ] LLM integration for answer generation
- [ ] Source citations
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Retrieval evaluation harness
- [ ] Semantic chunking
- [ ] Reranking

## Status

Currently under development.