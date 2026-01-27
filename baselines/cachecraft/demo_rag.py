"""
Demo script for RAG module.

Shows how to use the RAG system for document retrieval.
"""

import argparse
import json
import time
from typing import List
import warnings

# Import RAG components
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from rag import SimpleRAG, SentenceChunker, SentenceTransformerEmbedder
from rag import VectorDocumentStore

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_basic_usage():
    """Basic usage demo with sample documents."""
    print_section("Basic RAG Usage Demo")

    # Sample documents
    docs = [
        {
            "id": "1",
            "title": "Python Programming",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation."
        },
        {
            "id": "2",
            "title": "Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Key algorithms include supervised learning, unsupervised learning, and reinforcement learning. Common applications include image recognition, natural language processing, and recommendation systems."
        },
        {
            "id": "3",
            "title": "Deep Learning",
            "text": "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from a lamppost."
        },
        {
            "id": "4",
            "title": "Neural Networks",
            "text": "Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. Neural networks can learn to represent complex patterns and make predictions about the world."
        },
        {
            "id": "5",
            "title": "Data Science",
            "text": "Data science combines statistics, computer science, and domain expertise to extract insights from data. Data scientists use various tools and techniques to collect, clean, analyze, and visualize data. Key skills include programming, statistics, machine learning, and data visualization."
        }
    ]

    # Initialize RAG
    print("Initializing RAG system...")
    rag = SimpleRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=100,
        chunker_type="sentence",
        use_faiss=True
    )

    # Add documents
    print("Adding documents...")
    rag.add_documents(docs)

    # Show stats
    stats = rag.get_stats()
    print(f"\nStore Statistics:")
    print(f"  - Chunks: {stats['num_chunks']}")
    print(f"  - Embedding dimension: {stats['embedding_dim']}")
    print(f"  - Using FAISS: {stats['using_faiss']}")
    print(f"  - Memory usage: {stats['embeddings_memory_mb']:.2f} MB")

    # Test queries
    queries = [
        "What is Python programming?",
        "Explain machine learning algorithms",
        "How do neural networks work?",
        "What skills do data scientists need?",
        "What is deep learning used for?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = rag.retrieve(query, top_k=2, return_metadata=True)

        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.4f}")
            print(f"    Doc ID: {r['doc_id']}")
            print(f"    Text: {r['text'][:150]}...")


def demo_load_from_file(file_path: str, top_k: int, query: str):
    """Demo loading documents from JSON/JSONL file."""
    print_section(f"Loading Documents from File: {file_path}")

    if not file_path:
        print("No file path provided. Skipping file loading demo.")
        return

    rag = SimpleRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=200,
        use_faiss=True
    )

    # Load based on file extension
    start_time = time.time()
    if file_path.endswith('.jsonl'):
        rag.load_jsonl(file_path)
    else:
        rag.load_json(file_path)
    load_time = time.time() - start_time

    # Show stats
    stats = rag.get_stats()
    print(f"\nDocument Statistics:")
    print(f"  - Loading time: {load_time:.2f} seconds")
    print(f"  - Total chunks: {stats['num_chunks']}")
    print(f"  - Embedding dimension: {stats['embedding_dim']}")
    print(f"  - Memory usage: {stats['embeddings_memory_mb']:.2f} MB")

    # Test retrieval
    print(f"\nTesting retrieval performance...")
    query = query or "What is the main topic of these documents?"

    # Warm-up
    rag.retrieve(query, top_k=1)

    # Measure retrieval time
    start_time = time.time()
    results = rag.retrieve(query, top_k=top_k, return_metadata=True)
    retrieval_time = time.time() - start_time

    print(f"\nQuery: {query}")
    print(f"Retrieval time: {retrieval_time*1000:.2f} ms")
    print(f"Results returned: {len(results)}")

    print("\nTop results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f}")
        print(f"    Doc ID: {r['doc_id']}")
        if 'title' in r['metadata']:
            print(f"    Title: {r['metadata']['title']}")
        print(f"    Text preview: {r['text'][:200]}...")


def demo_performance_benchmark():
    """Demo with performance benchmarking."""
    print_section("Performance Benchmark")

    # Generate synthetic documents
    print("Generating 100 synthetic documents...")
    docs = []
    for i in range(100):
        docs.append({
            "id": str(i),
            "title": f"Document {i}",
            "text": f"This is document number {i}. " * 20 +  # Repeat to make it longer
                   f"It contains various information about topic {i % 10}. " +
                   f"Keywords: keyword_{i%5}, topic_{i%10}, section_{i%3}. " * 5
        })

    # Initialize RAG
    print("\nInitializing RAG system...")
    rag = SimpleRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=100,
        use_faiss=True
    )

    # Add documents and time it
    start_time = time.time()
    rag.add_documents(docs)
    add_time = time.time() - start_time

    stats = rag.get_stats()
    print(f"\nIndexing Statistics:")
    print(f"  - Documents: {len(docs)}")
    print(f"  - Chunks generated: {stats['num_chunks']}")
    print(f"  - Indexing time: {add_time:.2f} seconds")
    print(f"  - Chunks per second: {stats['num_chunks']/add_time:.1f}")
    print(f"  - Memory usage: {stats['embeddings_memory_mb']:.2f} MB")

    # Test retrieval with multiple queries
    test_queries = [
        "What is document 10 about?",
        "Tell me about keyword_3",
        "What information is in topic_5?",
        "Find documents about section_1",
        "What can you tell me about document 50?"
    ]

    print("\n" + "-"*60)
    print("Retrieval Benchmark (5 queries)")
    print("-"*60)

    total_retrieval_time = 0
    for query in test_queries:
        start_time = time.time()
        results = rag.retrieve(query, top_k=5)
        retrieval_time = time.time() - start_time
        total_retrieval_time += retrieval_time
        print(f"  Query: {query[:50]:50s} | Time: {retrieval_time*1000:6.2f} ms | Results: {len(results)}")

    print(f"\nAverage retrieval time: {(total_retrieval_time/len(test_queries))*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="RAG Module Demo")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "file", "benchmark", "all"],
                        help="Demo mode to run")
    parser.add_argument("--document_path", type=str, default="",
                        help="Path to JSON/JSONL document file")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-Transformers model name")
    parser.add_argument("--chunk_size", type=int, default=200,
                        help="Chunk size for document splitting")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--query", type=str, default="",
                        help="Query for retrieval test")

    args = parser.parse_args()

    if args.mode == "basic" or args.mode == "all":
        demo_basic_usage()

    if args.mode == "file" or args.mode == "all":
        demo_load_from_file(args.document_path, args.top_k, args.query)

    if args.mode == "benchmark" or args.mode == "all":
        demo_performance_benchmark()

    print_section("Demo Complete")


if __name__ == "__main__":
    main()
