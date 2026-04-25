"""
Supabase pgvector client — persistent vector storage replacing ChromaDB.
"""

import os
from supabase import create_client, Client

_client = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _client


def upsert_documents(ids, contents, embeddings, sources, chunk_indices, collection):
    client = get_supabase()
    rows = [
        {
            "id": id_,
            "content": content,
            "embedding": emb,
            "source": source,
            "chunk_index": idx,
            "collection": collection,
        }
        for id_, content, emb, source, idx in zip(
            ids, contents, embeddings, sources, chunk_indices
        )
    ]
    for i in range(0, len(rows), 50):
        client.table("documents").upsert(rows[i : i + 50]).execute()


def similarity_search(query_embedding, collection, n_results=5):
    client = get_supabase()
    result = client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "collection_name": collection,
            "match_count": n_results,
        },
    ).execute()
    return result.data or []


def count_collection(collection):
    try:
        client = get_supabase()
        result = (
            client.table("documents")
            .select("id", count="exact")
            .eq("collection", collection)
            .execute()
        )
        return result.count or 0
    except Exception:
        return 0


def delete_collection(collection):
    try:
        client = get_supabase()
        client.table("documents").delete().eq("collection", collection).execute()
    except Exception:
        pass
