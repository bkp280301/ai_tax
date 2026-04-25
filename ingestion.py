"""
Parses PDFs/docs into chunks and stores them in ChromaDB.
Two collections: 'irs_regulations' and 'user_documents'.
"""

import os
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = Path(__file__).parent / "chroma_db"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks

_client = None
_embed_fn = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embed_fn


def _get_collection(name: str):
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


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
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]  # drop tiny trailing chunks


def _doc_id(file_path: str, chunk_index: int) -> str:
    file_hash = hashlib.md5(Path(file_path).name.encode()).hexdigest()[:8]
    return f"{file_hash}_chunk_{chunk_index}"


def ingest_file(file_path: str, collection_name: str) -> int:
    """
    Ingest a single file into the given ChromaDB collection.
    Returns the number of chunks stored.
    Skips chunks already present (idempotent by doc_id).
    """
    text = _extract_text(file_path)
    chunks = _chunk_text(text)
    collection = _get_collection(collection_name)

    ids = [_doc_id(file_path, i) for i in range(len(chunks))]
    metadatas = [
        {"source": Path(file_path).name, "chunk": i}
        for i in range(len(chunks))
    ]

    # ChromaDB upsert is idempotent
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def ingest_directory(dir_path: str, collection_name: str) -> dict:
    """
    Ingest all supported files in a directory.
    Returns {filename: chunk_count}.
    """
    supported = {".pdf", ".txt", ".md"}
    results = {}
    for file in Path(dir_path).iterdir():
        if file.suffix.lower() in supported:
            count = ingest_file(str(file), collection_name)
            results[file.name] = count
    return results


def delete_collection(name: str) -> None:
    """Delete a ChromaDB collection (used to wipe per-session user data)."""
    try:
        _get_client().delete_collection(name)
    except Exception:
        pass


def collection_stats(user_col: str = "user_documents",
                     prior_col: str = "prior_year_returns") -> dict:
    """Return document counts for IRS base + the caller's session collections."""
    client = _get_client()
    stats = {}
    for label, name in [("irs_regulations", "irs_regulations"),
                         ("user_documents", user_col),
                         ("prior_year_returns", prior_col)]:
        try:
            col = client.get_collection(name, embedding_function=_get_embed_fn())
            stats[label] = col.count()
        except Exception:
            stats[label] = 0
    return stats


if __name__ == "__main__":
    # One-time IRS loader: python ingestion.py
    irs_dir = Path(__file__).parent / "irs_docs"
    print("Loading IRS documents...")
    results = ingest_directory(str(irs_dir), "irs_regulations")
    if results:
        for fname, count in results.items():
            print(f"  {fname}: {count} chunks")
    else:
        print("  No files found in irs_docs/")
    print("\nCollection stats:", collection_stats())
