"""
Parses PDFs/docs into chunks, generates embeddings, stores in Supabase pgvector.
"""

import hashlib
from pathlib import Path

import fitz
from sentence_transformers import SentenceTransformer

from supabase_db import upsert_documents, count_collection
from supabase_db import delete_collection  # re-exported for app.py

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, convert_to_numpy=True).tolist()


def _extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext in (".txt", ".md"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    elif ext in (".csv", ".xlsx", ".xls"):
        from transaction_parser import parse_transactions, summarize_transactions, transactions_to_text
        df = parse_transactions(file_path)
        summary = summarize_transactions(df)
        return transactions_to_text(df, summary)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def _doc_id(file_path: str, chunk_index: int) -> str:
    file_hash = hashlib.md5(Path(file_path).name.encode()).hexdigest()[:8]
    return f"{file_hash}_chunk_{chunk_index}"


def ingest_file(file_path: str, collection_name: str) -> int:
    text = _extract_text(file_path)
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    ids = [_doc_id(file_path, i) for i in range(len(chunks))]
    sources = [Path(file_path).name] * len(chunks)
    indices = list(range(len(chunks)))
    embeddings = _embed(chunks)
    upsert_documents(ids, chunks, embeddings, sources, indices, collection_name)
    return len(chunks)


def ingest_directory(dir_path: str, collection_name: str) -> dict:
    supported = {".pdf", ".txt", ".md"}
    results = {}
    for file in Path(dir_path).iterdir():
        if file.suffix.lower() in supported:
            results[file.name] = ingest_file(str(file), collection_name)
    return results


def collection_stats(user_col: str = "user_documents",
                     prior_col: str = "prior_year_returns") -> dict:
    return {
        "irs_regulations": count_collection("irs_regulations"),
        "user_documents":  count_collection(user_col),
        "prior_year_returns": count_collection(prior_col),
    }


if __name__ == "__main__":
    irs_dir = Path(__file__).parent / "irs_docs"
    print("Loading IRS documents into Supabase...")
    results = ingest_directory(str(irs_dir), "irs_regulations")
    for fname, count in results.items():
        print(f"  {fname}: {count} chunks")
    print("\nCollection stats:", collection_stats())
