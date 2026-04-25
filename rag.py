"""
Retrieval layer using Supabase pgvector similarity search.
"""

from ingestion import _embed
from supabase_db import similarity_search


def retrieve(query: str, n_irs: int = 4, n_user: int = 4,
             user_col: str = "user_documents") -> dict:
    emb = _embed([query])[0]
    return {
        "irs":  similarity_search(emb, "irs_regulations", n_irs),
        "user": similarity_search(emb, user_col, n_user),
    }


def format_context(chunks: dict) -> str:
    parts = []
    if chunks["irs"]:
        parts.append("=== IRS Regulations & Guidelines ===")
        for c in chunks["irs"]:
            parts.append(f"[Source: {c['source']}]\n{c['content']}")
    if chunks["user"]:
        parts.append("\n=== Your Tax Documents ===")
        for c in chunks["user"]:
            parts.append(f"[Source: {c['source']}]\n{c['content']}")
    return "\n\n".join(parts) if parts else "No relevant documents found."


def retrieve_and_format(query: str, user_col: str = "user_documents") -> str:
    return format_context(retrieve(query, user_col=user_col))


def retrieve_and_format_prior_year(query: str,
                                   prior_col: str = "prior_year_returns") -> str:
    emb = _embed([query])[0]
    chunks = similarity_search(emb, prior_col, n_results=6)
    if not chunks:
        return "No prior year filings uploaded yet."
    parts = ["=== Prior Year Tax Filings ==="]
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['content']}")
    return "\n\n".join(parts)
