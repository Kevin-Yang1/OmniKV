"""
Chunking strategies for RAG.

Provides sentence-based and token-based document chunking.
"""

import sys
import os
from typing import List
import re

# Add LongBench to path to reuse their splitter
LONGBENCH_PATH = os.path.join(os.path.dirname(__file__), '../../../benchmark/long_bench/retrieval')
if os.path.exists(LONGBENCH_PATH):
    sys.path.insert(0, LONGBENCH_PATH)

# Try to import LongBench's splitter
try:
    from splitter import split_long_sentence, get_word_len, regex
    HAS_LONGBENCH_SPLITTER = True
except ImportError:
    HAS_LONGBENCH_SPLITTER = False

    # Fallback implementation if LongBench splitter is not available
    regex = r'([。？！；\n.!?;]\s*)'

    def get_word_list(s1):
        """Separate sentences by word, Chinese by word, English by word, numbers by space."""
        regEx = re.compile(r'[\W]')
        res = re.compile(r"([\u4e00-\u9fa5])")    # [\u4e00-\u9fa5] for Chinese

        p1 = regEx.split(s1.lower())
        str1_list = []
        for s in p1:
            if res.split(s) is None:
                str1_list.append(s)
            else:
                ret = res.split(s)
                for ch in ret:
                    str1_list.append(ch)

        list_word1 = [w for w in str1_list if len(w.strip()) > 0]
        return list_word1

    def get_word_len(s1):
        return len(get_word_list(s1))

    def split_long_sentence(sentence, regex, chunk_size=200, filename='Unknown'):
        """Split long sentence into chunks based on regex."""
        chunks = []
        sentences = re.split(regex, sentence)
        current_chunk = ""
        for s in sentences:
            if current_chunk and get_word_len(current_chunk) + get_word_len(s) <= chunk_size:
                current_chunk += ' ' if s == '' else s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_len = get_word_len(current_chunk)
                    if current_len > chunk_size * 1.5:
                        print(f"\n{filename}-{len(chunks)-1} Chunk size: {current_len}")
                current_chunk = s

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class SentenceChunker:
    """
    Sentence-based chunker using LongBench's splitter.

    Splits text into chunks based on sentence boundaries (。？！；\n.!?;).
    """

    def __init__(self, chunk_size: int = 200):
        """
        Initialize sentence chunker.

        Args:
            chunk_size: Maximum words/tokens per chunk
        """
        self.chunk_size = chunk_size
        self.regex = regex

    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = split_long_sentence(
            text,
            self.regex,
            chunk_size=self.chunk_size,
            filename='Document'
        )

        # Filter out empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]

        return chunks


class FixedTokenChunker:
    """
    Token-based chunker using a tokenizer.

    Splits text into chunks with fixed token counts, with optional overlap.
    """

    def __init__(self, tokenizer, max_tokens: int = 256, overlap: int = 0):
        """
        Initialize token chunker.

        Args:
            tokenizer: Tokenizer instance (e.g., from transformers)
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks with fixed token counts.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Encode text to tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())
                chunk_idx += 1

            # Move start position with overlap
            if end >= len(tokens):
                break
            start = end - self.overlap

        return chunks


class RecursiveCharacterChunker:
    """
    Recursive character chunker (similar to LangChain's RecursiveCharacterTextSplitter).

    Tries different separators in order of priority to split text.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators=None):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters
            separators: List of separators to try (in order of priority)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            # Default separators (ordered by priority)
            self.separators = ["\n\n", "\n", ". ", "。", "!", "！", "?", "？", ";", "；", " ", ""]
        else:
            self.separators = separators

    def chunk(self, text: str) -> List[str]:
        """
        Split text recursively using separators.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        return self._recursive_split(text, separators=self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""

                for part in parts:
                    if not part:
                        continue

                    if not current_chunk:
                        current_chunk = part
                    elif len(current_chunk) + len(separator) + len(part) <= self.chunk_size:
                        current_chunk += separator + part
                    else:
                        # Current chunk is full, save it
                        if current_chunk:
                            chunks.append(current_chunk)

                        # Handle overlap
                        if self.chunk_overlap > 0 and len(chunks) > 0:
                            overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                            current_chunk = current_chunk[overlap_start:] + separator + part
                        else:
                            current_chunk = part

                # Don't forget the last chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # If we still have chunks that are too large, recurse
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size and len(separators) > 1:
                        # Try again with remaining separators
                        sub_chunks = self._recursive_split(
                            chunk,
                            separators=separators[separators.index(separator) + 1:]
                        )
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)

                return final_chunks

        # No separator worked, split by character
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
