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


### MMR Retrieval Experiment

Implemented Maximal Marginal Relevance (MMR) to investigate the trade-off between retrieval relevance and diversity.

Compared standard cosine-similarity retrieval against MMR with lambda values of 0.7, 0.5, and 0.3, using 8 evaluation questions and two research papers.

**Results:**
- Standard similarity search achieved 100% Hit Rate@10 and 100% Recall@10.
- MMR with lambda = 0.5 achieved the same Hit Rate@10 and Recall@10, with MRR improving from 0.6615 to 0.6823.
- Lower lambda values did not consistently improve retrieval performance.

**Conclusion:** MMR can improve ranking quality in some cases, but its effectiveness depends on the relevance-diversity trade-off. Standard similarity search remains the default retrieval strategy.