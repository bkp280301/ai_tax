"""
Retrieval layer over ChromaDB.
Queries IRS regulations and session-specific user document collections.
"""

from ingestion import _get_collection


def _query_collection(collection_name: str, query: str, n_results: int) -> list[dict]:
    try:
        col = _get_collection(collection_name)
        if col.count() == 0:
            return []
        results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"text": doc, "source": meta["source"], "score": 1 - dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
    except Exception:
        return []


def retrieve(query: str, n_irs: int = 4, n_user: int = 4,
             user_col: str = "user_documents") -> dict:
    return {
        "irs": _query_collection("irs_regulations", query, n_irs),
        "user": _query_collection(user_col, query, n_user),
    }


def format_context(chunks: dict) -> str:
    parts = []
    if chunks["irs"]:
        parts.append("=== IRS Regulations & Guidelines ===")
        for c in chunks["irs"]:
            parts.append(f"[Source: {c['source']}]\n{c['text']}")
    if chunks["user"]:
        parts.append("\n=== Your Tax Documents ===")
        for c in chunks["user"]:
            parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts) if parts else "No relevant documents found."


def retrieve_and_format(query: str, user_col: str = "user_documents") -> str:
    return format_context(retrieve(query, user_col=user_col))


def retrieve_prior_year(query: str, n_results: int = 6,
                        prior_col: str = "prior_year_returns") -> list[dict]:
    return _query_collection(prior_col, query, n_results)


def retrieve_and_format_prior_year(query: str,
                                   prior_col: str = "prior_year_returns") -> str:
    chunks = retrieve_prior_year(query, prior_col=prior_col)
    if not chunks:
        return "No prior year filings uploaded yet."
    parts = ["=== Prior Year Tax Filings ==="]
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)
