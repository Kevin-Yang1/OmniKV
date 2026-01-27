"""
RAG (Retrieval-Augmented Generation) module for Cache-Craft.

Provides:
- Chunker: Document chunking strategies (sentence-based, token-based)
- Embedder: Sentence-Transformers wrapper for dense embeddings
- DocumentStore: Vector store with FAISS indexing
- SimpleRAG: Simplified RAG wrapper for quick setup

Usage:
    from baselines.cachecraft.rag import SimpleRAG

    # Initialize RAG
    rag = SimpleRAG(embedding_model="sentence-transformers/all-MiniLM-L6-v2", chunk_size=200)
    rag.load_json("documents.json")

    # Retrieve
    results = rag.retrieve("What is the capital of France?", top_k=5)
"""

from .chunker import SentenceChunker, FixedTokenChunker
from .embedder import SentenceTransformerEmbedder
from .document_store import VectorDocumentStore
from .rag_wrapper import SimpleRAG

__all__ = [
    'SentenceChunker',
    'FixedTokenChunker',
    'SentenceTransformerEmbedder',
    'VectorDocumentStore',
    'SimpleRAG',
]
