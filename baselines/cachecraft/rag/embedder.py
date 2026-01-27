"""
Embedder module for generating dense text embeddings.

Uses Sentence-Transformers for efficient batch embedding with GPU support.
"""

import numpy as np
from typing import List, Union
import warnings


class SentenceTransformerEmbedder:
    """
    Sentence-Transformers wrapper for generating embeddings.

    Supports batch processing, GPU acceleration, and automatic normalization.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedder.

        Args:
            model_name: Name of the Sentence-Transformers model
                       (e.g., "sentence-transformers/all-MiniLM-L6-v2",
                        "intfloat/e5-large-v2", "BAAI/bge-m3")
            device: Device to run on ("cuda", "cpu", or None for auto-detection)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings (for cosine similarity)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )

        # Auto-detect device if not specified
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        print(f"[Embedder] Loading model '{model_name}' on device '{device}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings
            show_progress: Whether to show progress bar

        Returns:
            Embeddings as numpy array: [num_texts, embedding_dim]
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype='float32')

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device
        )

        return embeddings

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings in explicit batches.

        Args:
            texts: List of text strings
            batch_size: Batch size (defaults to self.batch_size)

        Returns:
            List of embedding arrays, one per batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.embed(batch_texts)
            embeddings_list.append(embeddings)

        return embeddings_list

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class OpenAIEmbedder:
    """
    Alternative embedder using OpenAI's API.

    Useful for production deployments without local GPU resources.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            model: Model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
            batch_size: Batch size for API calls
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required. "
                "Install it with: pip install openai"
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

        # Get embedding dimension based on model
        self.embedding_dim = self._get_embedding_dim(model)
        print(f"[OpenAIEmbedder] Using model '{model}'. Embedding dimension: {self.embedding_dim}")

    def _get_embedding_dim(self, model: str) -> int:
        """Get embedding dimension for a given model."""
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(model, 1536)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Embeddings as numpy array: [num_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype='float32')

        embeddings_list = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.model
            )

            batch_embeddings = np.array(
                [item.embedding for item in response.data],
                dtype='float32'
            )
            embeddings_list.append(batch_embeddings)

        return np.vstack(embeddings_list)
