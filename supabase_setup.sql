-- Run this entire file in Supabase → SQL Editor → New Query

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Documents table (replaces ChromaDB)
create table if not exists documents (
  id          text primary key,
  content     text        not null,
  embedding   vector(384) not null,
  source      text,
  chunk_index int,
  collection  text        not null
);

-- 3. Indexes
create index if not exists documents_collection_idx
  on documents (collection);

create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Similarity search function
create or replace function match_documents(
  query_embedding vector(384),
  collection_name text,
  match_count     int default 5
)
returns table(id text, content text, source text, similarity float)
language sql stable
as $$
  select id, content, source,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where collection = collection_name
  order by embedding <=> query_embedding
  limit match_count;
$$;
