"""Retrieval over document text: chunking, embeddings, indexing, and
hybrid (pgvector + Postgres full-text) search with page-level citations.

Pure/offline pieces (chunking, the hashing embedder) have no DB or
network dependency; indexing and search run against Postgres.
"""
