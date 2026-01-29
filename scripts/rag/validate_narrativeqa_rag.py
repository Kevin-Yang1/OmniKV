"""
验证 NarrativeQA RAG 数据集质量

检查:
1. 数据完整性
2. Chunk 检索相关性
3. 格式正确性
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def validate_rag_dataset(rag_file: str, original_file: str):
    """验证 RAG 数据集"""
    
    print("="*60)
    print("NarrativeQA RAG Dataset Validation")
    print("="*60)
    
    # 加载数据
    print("\n[1/4] Loading datasets...")
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"  RAG samples: {len(rag_data)}")
    print(f"  Original samples: {len(original_data)}")
    
    # 验证数据完整性
    print("\n[2/4] Validating data integrity...")
    assert len(rag_data) == len(original_data), "Sample count mismatch!"
    
    for i, (rag_sample, orig_sample) in enumerate(zip(rag_data, original_data)):
        assert rag_sample['_id'] == orig_sample['_id'], f"ID mismatch at index {i}"
        assert rag_sample['input'] == orig_sample['input'], f"Question mismatch at index {i}"
        assert rag_sample['answers'] == orig_sample['answers'], f"Answer mismatch at index {i}"
    
    print("  ✓ All samples match original data")
    
    # 分析检索结果
    print("\n[3/4] Analyzing retrieval results...")
    k_values = [len(sample['context']) for sample in rag_data]
    k_counter = Counter(k_values)
    print(f"  Chunks per sample: {dict(k_counter)}")
    
    # 长度统计
    orig_lengths = [s['length'] for s in original_data]
    rag_lengths = [s['length'] for s in rag_data]
    
    print(f"\n  Original context length:")
    print(f"    Mean: {sum(orig_lengths)/len(orig_lengths):.0f}")
    print(f"    Min: {min(orig_lengths)}, Max: {max(orig_lengths)}")
    
    print(f"\n  RAG context length (k chunks):")
    print(f"    Mean: {sum(rag_lengths)/len(rag_lengths):.0f}")
    print(f"    Min: {min(rag_lengths)}, Max: {max(rag_lengths)}")
    
    compression_ratio = sum(orig_lengths) / sum(rag_lengths)
    print(f"\n  Compression ratio: {compression_ratio:.2f}x")
    
    # 检索分数分析
    print("\n[4/4] Analyzing retrieval scores...")
    all_scores = []
    for sample in rag_data:
        all_scores.extend(sample['metadata']['chunk_scores'])
    
    print(f"  Mean score: {sum(all_scores)/len(all_scores):.4f}")
    print(f"  Min score: {min(all_scores):.4f}")
    print(f"  Max score: {max(all_scores):.4f}")
    
    # 抽样检查
    print("\n[Sample Check] First sample:")
    sample = rag_data[0]
    print(f"  Question: {sample['input'][:80]}...")
    print(f"  Answer: {sample['answers'][0][:80]}...")
    print(f"  Chunks: {len(sample['context'])}")
    print(f"  Top-3 scores: {sample['metadata']['chunk_scores'][:3]}")
    print(f"  First chunk preview: {sample['context'][0][:100]}...")
    
    print("\n" + "="*60)
    print("✅ Validation completed successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate RAG dataset')
    parser.add_argument('--rag', type=str, required=True, help='RAG dataset file')
    parser.add_argument('--original', type=str, required=True, help='Original dataset file')
    
    args = parser.parse_args()
    validate_rag_dataset(args.rag, args.original)


if __name__ == '__main__':
    main()
