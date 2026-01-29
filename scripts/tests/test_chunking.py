"""测试递归分割策略"""

import sys
sys.path.insert(0, '/NV1/ykw/projects/OmniKV')
from create_narrativeqa_rag import NarrativeQARAGConverter

# 测试文本
test_text = """
This is the first paragraph. It contains multiple sentences. This is another sentence in the first paragraph.

This is the second paragraph. It has different content. The second paragraph continues here.

This is a very long paragraph that will definitely exceed the chunk size limit so we can test how the recursive splitting works when a single paragraph is too large for a single chunk. It should split at sentence boundaries first, and if that doesn't work, it will try other separators.

Final short paragraph.
""".strip()

print("="*60)
print("Testing Recursive Character Splitting")
print("="*60)

# 初始化 converter (不加载模型，只测试 chunking)
class TestConverter:
    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    # 从 NarrativeQARAGConverter 复制方法
    chunk_text = NarrativeQARAGConverter.chunk_text
    _recursive_split = NarrativeQARAGConverter._recursive_split
    _force_split = NarrativeQARAGConverter._force_split

converter = TestConverter(chunk_size=200, overlap=50)

print(f"\nTest text length: {len(test_text)} chars")
print(f"Chunk size: {converter.chunk_size}")
print(f"Overlap: {converter.overlap}")

chunks = converter.chunk_text(test_text)

print(f"\nGenerated {len(chunks)} chunks:")
print("="*60)

for i, chunk in enumerate(chunks):
    print(f"\n[Chunk {i+1}] ({chunk['start_pos']}-{chunk['end_pos']}, {len(chunk['text'])} chars)")
    print("-"*60)
    print(chunk['text'][:150] + ('...' if len(chunk['text']) > 150 else ''))
    
    # 检查重叠
    if i > 0:
        prev_text = chunks[i-1]['text']
        curr_text = chunk['text']
        # 检查是否有重叠内容
        overlap_found = False
        for j in range(min(len(prev_text), len(curr_text))):
            if prev_text[-j:] == curr_text[:j]:
                if j > 10:  # 至少10字符重叠才算
                    overlap_found = True
                    print(f"  [Overlap with prev: ~{j} chars]")
                    break

print("\n" + "="*60)
print("✅ Recursive splitting test completed!")
