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


def retrieve_all_user_docs(user_col: str, n_per_query: int = 12) -> str:
    """
    Run multiple targeted queries to pull comprehensive user financial data.
    Deduplicates by content so each chunk appears only once.
    """
    queries = [
        "W-2 wages income salary payroll employer Box 1 federal withholding",
        "1099 self employment consulting freelance income schedule C",
        "deductions mortgage interest charitable contributions medical dental",
        "401k HSA IRA retirement contributions pre-tax savings",
        "childcare dependent care child education 529 expenses",
        "business expenses office equipment travel meals software advertising",
        "tax withheld estimated payments quarterly filing status dependents",
    ]

    seen_content = set()
    user_chunks = []
    irs_chunks = []

    for q in queries:
        emb = _embed([q])[0]
        u_results = similarity_search(emb, user_col, n_per_query)
        i_results = similarity_search(emb, "irs_regulations", 2)
        for c in u_results:
            key = c["content"][:120]
            if key not in seen_content:
                seen_content.add(key)
                user_chunks.append(c)
        for c in i_results:
            key = c["content"][:120]
            if key not in seen_content:
                seen_content.add(key)
                irs_chunks.append(c)

    parts = []
    if irs_chunks:
        parts.append("=== IRS Regulations & Tax Law ===")
        for c in irs_chunks[:8]:
            parts.append(f"[Source: {c['source']}]\n{c['content']}")

    if user_chunks:
        parts.append("\n=== USER FINANCIAL DOCUMENTS (W-2s, 1099s, Transactions, Tax Profile) ===")
        for c in user_chunks:
            parts.append(f"[Source: {c['source']}]\n{c['content']}")
    else:
        parts.append("\n=== USER FINANCIAL DOCUMENTS ===\nNo user documents found in this session.")

    return "\n\n".join(parts)


def retrieve_all_prior_docs(prior_col: str, n_per_query: int = 8) -> str:
    """Pull comprehensive prior-year data with multiple queries."""
    queries = [
        "prior year income W-2 wages 1099 AGI adjusted gross income",
        "prior year deductions mortgage charitable medical retirement",
        "prior year tax paid refund federal withholding estimated payments",
    ]
    seen = set()
    chunks = []
    for q in queries:
        emb = _embed([q])[0]
        for c in similarity_search(emb, prior_col, n_per_query):
            key = c["content"][:120]
            if key not in seen:
                seen.add(key)
                chunks.append(c)

    if not chunks:
        return "No prior year filings uploaded yet."
    parts = ["=== Prior Year Tax Filings ==="]
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['content']}")
    return "\n\n".join(parts)


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
