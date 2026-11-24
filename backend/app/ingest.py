from pathlib import Path
from ingestion import ingest_document

documents_folder = Path("documents")

pdf_files = list(documents_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
else:
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        ingest_document(str(pdf_path))