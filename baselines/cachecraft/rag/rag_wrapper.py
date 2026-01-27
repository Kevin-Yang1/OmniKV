"""
Simplified RAG wrapper for quick setup and easy usage.

Provides SimpleRAG class that handles document loading, chunking,
embedding, and retrieval with minimal configuration.
"""

import os
import json
from typing import List, Dict
from .document_store import VectorDocumentStore
from .embedder import SentenceTransformerEmbedder
from .chunker import SentenceChunker, FixedTokenChunker


class SimpleRAG:
    """
    Simplified RAG wrapper for quick setup.

    Handles document loading, chunking, embedding, and retrieval with minimal configuration.

    Usage:
        rag = SimpleRAG(embedding_model="sentence-transformers/all-MiniLM-L6-v2", chunk_size=200)
        rag.load_json("documents.json")
        results = rag.retrieve("What is the capital of France?", top_k=5)
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 200,
        chunker_type: str = "sentence",
        device: str = None,
        batch_size: int = 32,
        use_faiss: bool = True,
        index_type: str = "flat"
    ):
        """
        Initialize SimpleRAG.

        Args:
            embedding_model: Name of the Sentence-Transformers model
            chunk_size: Maximum size for chunks (words/tokens depending on chunker)
            chunker_type: Type of chunker ("sentence", "token", "recursive")
            device: Device for embedding model ("cuda", "cpu", or None for auto)
            batch_size: Batch size for embedding
            use_faiss: Whether to use FAISS for indexing
            index_type: FAISS index type ("flat", "ivf", "hnsw")
        """
        # Initialize embedder
        self.embedder = SentenceTransformerEmbedder(
            model_name=embedding_model,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True
        )

        # Initialize chunker
        if chunker_type == "sentence":
            self.chunker = SentenceChunker(chunk_size=chunk_size)
        elif chunker_type == "token":
            # For token-based chunking, tokenizer must be provided separately
            # or we'll use sentence chunking as fallback
            warnings.warn("Token-based chunker requires a tokenizer. Using sentence chunker instead.")
            self.chunker = SentenceChunker(chunk_size=chunk_size)
        elif chunker_type == "recursive":
            from .chunker import RecursiveCharacterChunker
            self.chunker = RecursiveCharacterChunker(chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown chunker_type: {chunker_type}")

        # Initialize document store
        self.store = VectorDocumentStore(
            embedder=self.embedder,
            chunker=self.chunker,
            use_faiss=use_faiss,
            index_type=index_type
        )

        print(f"[SimpleRAG] Initialized with model='{embedding_model}', chunker='{chunker_type}', chunk_size={chunk_size}")

    def load_jsonl(self, file_path: str, text_field: str = "text") -> None:
        """
        Load documents from JSONL file.

        Args:
            file_path: Path to JSONL file (one JSON object per line)
            text_field: Field name containing the document text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    # Ensure text field exists
                    if text_field not in doc:
                        doc['text'] = str(doc)
                    docs.append(doc)
                except json.JSONDecodeError:
                    continue

        print(f"[RAG] Loaded {len(docs)} documents from {file_path}")
        self.store.add_documents(docs)

    def load_json(self, file_path: str, text_field: str = "text") -> None:
        """
        Load documents from JSON file (array of objects).

        Args:
            file_path: Path to JSON file containing array of document objects
            text_field: Field name containing the document text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)

        # Ensure all docs have text field
        for doc in docs:
            if text_field not in doc:
                doc['text'] = str(doc)

        print(f"[RAG] Loaded {len(docs)} documents from {file_path}")
        self.store.add_documents(docs)

    def add_documents(self, docs: List[Dict]) -> None:
        """
        Add documents directly.

        Args:
            docs: List of document dictionaries, each with 'text' field
        """
        self.store.add_documents(docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = False
    ) -> List[str]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            return_metadata: Whether to return full result objects (with scores, etc.)

        Returns:
            If return_metadata=False: List of text strings
            If return_metadata=True: List of result dictionaries with metadata
        """
        results = self.store.retrieve(query, top_k=top_k, include_metadata=True)

        if return_metadata:
            return results
        else:
            return [r['text'] for r in results]

    def get_stats(self) -> Dict:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with statistics
        """
        return self.store.get_stats()


class TokenRAG(SimpleRAG):
    """
    RAG wrapper that uses token-based chunking.

    Requires a tokenizer to be provided.
    """

    def __init__(
        self,
        tokenizer,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 256,
        overlap: int = 0,
        device: str = None,
        batch_size: int = 32,
        use_faiss: bool = True
    ):
        """
        Initialize TokenRAG.

        Args:
            tokenizer: Tokenizer instance (e.g., from transformers.AutoTokenizer)
            embedding_model: Name of the Sentence-Transformers model
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            device: Device for embedding model
            batch_size: Batch size for embedding
            use_faiss: Whether to use FAISS for indexing
        """
        # Initialize embedder
        self.embedder = SentenceTransformerEmbedder(
            model_name=embedding_model,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True
        )

        # Initialize token-based chunker
        self.chunker = FixedTokenChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap=overlap
        )

        # Initialize document store
        self.store = VectorDocumentStore(
            embedder=self.embedder,
            chunker=self.chunker,
            use_faiss=use_faiss
        )

        print(f"[TokenRAG] Initialized with model='{embedding_model}', max_tokens={max_tokens}, overlap={overlap}")
