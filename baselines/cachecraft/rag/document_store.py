"""
Document store with FAISS indexing for efficient retrieval.

Manages document chunks, embeddings, and FAISS index for fast similarity search.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


class VectorDocumentStore:
    """
    Vector document store with FAISS indexing.

    Supports adding documents, building FAISS index, and efficient retrieval.
    """

    def __init__(
        self,
        embedder,
        chunker=None,
        use_faiss: bool = True,
        index_type: str = "flat"
    ):
        """
        Initialize document store.

        Args:
            embedder: Embedder instance (e.g., SentenceTransformerEmbedder)
            chunker: Chunker instance (e.g., SentenceChunker). If None, documents won't be chunked.
            use_faiss: Whether to use FAISS for indexing
            index_type: FAISS index type ("flat", "ivf", "hnsw")
        """
        self.embedder = embedder
        self.chunker = chunker
        self.use_faiss = use_faiss
        self.index_type = index_type

        # Storage
        self.chunks: List[Dict] = []      # {'text': str, 'doc_id': str, 'metadata': dict}
        self.embeddings: Optional[np.ndarray] = None  # [num_chunks, embedding_dim]
        self.faiss_index = None

        # FAISS index parameters
        self.embedding_dim = embedder.dimension

    def add_documents(self, docs: List[Dict], chunk_documents: bool = True) -> None:
        """
        Add documents to the store.

        Args:
            docs: List of documents. Each doc should have 'text' field.
                  Format: [{'id': str, 'text': str, 'title': str, ...}, ...]
            chunk_documents: Whether to chunk documents (requires chunker)
        """
        if not docs:
            warnings.warn("No documents provided to add_documents()")
            return

        all_chunks = []

        for doc in docs:
            text = doc.get('text', '')

            if not text or not text.strip():
                continue

            # Chunk document if chunker is provided
            if self.chunker and chunk_documents:
                chunks = self.chunker.chunk(text)
            else:
                chunks = [text]

            for chunk_text in chunks:
                if chunk_text.strip():
                    all_chunks.append({
                        'text': chunk_text.strip(),
                        'doc_id': doc.get('id', str(len(self.chunks) + len(all_chunks))),
                        'title': doc.get('title', ''),
                        'metadata': doc
                    })

        if not all_chunks:
            warnings.warn("No chunks generated from documents")
            return

        # Store chunks
        self.chunks.extend(all_chunks)
        chunk_texts = [c['text'] for c in all_chunks]

        print(f"[DocumentStore] Processing {len(chunk_texts)} chunks...")

        # Batch embed
        new_embeddings = self.embedder.embed(chunk_texts)

        # Merge embeddings
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Build/update FAISS index
        self._build_index()

    def _build_index(self) -> None:
        """Build or update FAISS index."""
        if not self.use_faiss or self.embeddings is None:
            return

        embedding_dim = self.embeddings.shape[1]

        if self.index_type == "flat":
            # IndexFlatIP: Inner Product (for normalized embeddings = cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index.add(self.embeddings.astype('float32'))
            print(f"[DocumentStore] FAISS (Flat) index built: {len(self.chunks)} chunks, dimension={embedding_dim}")

        elif self.index_type == "ivf":
            # IVF index with quantization
            try:
                nlist = min(100, len(self.chunks))  # Number of clusters
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)

                # Train index
                self.faiss_index.train(self.embeddings.astype('float32'))
                self.faiss_index.add(self.embeddings.astype('float32'))
                print(f"[DocumentStore] FAISS (IVF) index built: {len(self.chunks)} chunks, nlist={nlist}")
            except Exception as e:
                warnings.warn(f"Failed to build IVF index, falling back to flat: {e}")
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index.add(self.embeddings.astype('float32'))

        elif self.index_type == "hnsw":
            # HNSW index (memory-efficient)
            try:
                self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)
                self.faiss_index.add(self.embeddings.astype('float32'))
                print(f"[DocumentStore] FAISS (HNSW) index built: {len(self.chunks)} chunks")
            except Exception as e:
                warnings.warn(f"Failed to build HNSW index, falling back to flat: {e}")
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index.add(self.embeddings.astype('float32'))

        else:
            warnings.warn(f"Unknown index type '{self.index_type}', using flat")
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index.add(self.embeddings.astype('float32'))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_metadata: bool = False
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return
            include_metadata: Whether to include full document metadata

        Returns:
            List of results: [{'text': str, 'score': float, 'doc_id': str, ...}, ...]
        """
        if len(self.chunks) == 0:
            return []

        # Embed query
        query_emb = self.embedder.embed([query])[0]  # [embedding_dim]

        if self.use_faiss and self.faiss_index is not None:
            # FAISS search
            scores, indices = self.faiss_index.search(
                query_emb.reshape(1, -1).astype('float32'),
                min(top_k, len(self.chunks))
            )
        else:
            # Brute force search
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(-similarities)[:top_k]
            scores = similarities[top_indices].reshape(1, -1)
            indices = top_indices.reshape(1, -1)

        # Assemble results
        results = []
        seen_texts = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)

            # Deduplicate by text
            text_hash = hash(chunk['text'])
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            # Optionally include full metadata
            if not include_metadata:
                chunk.pop('metadata', None)

            results.append(chunk)

        return results

    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        print("[DocumentStore] Cleared all documents")

    def get_stats(self) -> Dict:
        """
        Get store statistics.

        Returns:
            Dictionary with store statistics
        """
        return {
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'using_faiss': self.use_faiss,
            'index_type': self.index_type,
            'embeddings_memory_mb': self.embeddings.nbytes / (1024 * 1024) if self.embeddings is not None else 0,
        }


# Import FAISS at module level with fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    warnings.warn(
        "FAISS is not installed. Install with: pip install faiss-cpu (or faiss-gpu for GPU). "
        "Falling back to brute-force search."
    )
