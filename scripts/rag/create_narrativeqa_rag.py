"""
NarrativeQA RAG Dataset Converter
将 narrativeqa.json 转换为 RAG 格式数据集

使用方法:
    python create_narrativeqa_rag.py --input datasets/longbench/narrativeqa.json \
                                     --output datasets/longbench/narrativeqa_rag \
                                     --k 10 \
                                     --chunk_size 1000 \
                                     --overlap 200
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class NarrativeQARAGConverter:
    def __init__(
        self,
        model_name: str = "nvidia/NV-Embed-v2",
        chunk_size: int = 1000,
        overlap: int = 200,
        device: str = "cuda"
    ):
        """
        初始化转换器
        
        Args:
            model_name: 嵌入模型名称
            chunk_size: chunk 大小（字符数）
            overlap: chunk 重叠大小
            device: 计算设备
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
        print("Model loaded successfully")
        
    def extract_unique_contexts(self, data: List[Dict]) -> Dict[str, Dict]:
        """
        提取唯一的 context（基于前100字符）
        
        Returns:
            {key: {'full_text': str, 'id': int, 'hash': str}}
        """
        print("\n[Step 1/5] Extracting unique contexts...")
        unique_contexts = {}
        
        for idx, sample in enumerate(tqdm(data, desc="Scanning contexts")):
            context_text = sample['context'][0]
            key = context_text[:100]  # 使用前100字符作为key
            
            if key not in unique_contexts:
                unique_contexts[key] = {
                    'full_text': context_text,
                    'id': len(unique_contexts),
                    'hash': key,
                    'sample_ids': []
                }
            
            unique_contexts[key]['sample_ids'].append(idx)
        
        print(f"Found {len(unique_contexts)} unique contexts")
        return unique_contexts
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        递归字符分割策略 (LangChain 风格)
        优先在段落、句子、词边界处切分，而不是硬截断
        
        Returns:
            List of {'text': str, 'start_pos': int, 'end_pos': int}
        """
        # 分隔符优先级：段落 > 句子 > 单词 > 字符
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ', '']
        
        chunks = self._recursive_split(text, separators, 0)
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str], start_pos: int) -> List[Dict]:
        """
        递归分割文本
        
        Args:
            text: 要分割的文本
            separators: 分隔符列表（优先级从高到低）
            start_pos: 当前文本在原文中的起始位置
        """
        chunks = []
        
        # 如果文本足够小，直接返回
        if len(text) <= self.chunk_size:
            if text.strip():  # 忽略空白文本
                chunks.append({
                    'text': text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(text)
                })
            return chunks
        
        # 选择当前分隔符
        if not separators:
            # 如果没有分隔符了，强制按 chunk_size 切分
            return self._force_split(text, start_pos)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 按当前分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            # 空分隔符表示按字符分割
            splits = list(text)
        
        # 合并小片段为合适大小的 chunks
        current_chunk = []
        current_len = 0
        
        for i, split in enumerate(splits):
            split_len = len(split)
            
            # 如果单个 split 就超过 chunk_size，递归处理
            if split_len > self.chunk_size:
                # 先保存当前积累的 chunk
                if current_chunk:
                    chunk_text = separator.join(current_chunk) if separator else ''.join(current_chunk)
                    if separator and i < len(splits) - 1:
                        chunk_text += separator  # 保留分隔符
                    
                    sub_chunks = self._recursive_split(
                        chunk_text, 
                        remaining_separators, 
                        start_pos
                    )
                    chunks.extend(sub_chunks)
                    start_pos += len(chunk_text)
                    current_chunk = []
                    current_len = 0
                
                # 递归处理超大 split
                sub_chunks = self._recursive_split(
                    split, 
                    remaining_separators, 
                    start_pos
                )
                chunks.extend(sub_chunks)
                start_pos += split_len
                
                # 加上分隔符
                if separator and i < len(splits) - 1:
                    start_pos += len(separator)
                
            else:
                # 检查是否需要开始新 chunk
                potential_len = current_len + split_len + (len(separator) if separator and current_chunk else 0)
                
                if potential_len > self.chunk_size and current_chunk:
                    # 保存当前 chunk
                    chunk_text = separator.join(current_chunk) if separator else ''.join(current_chunk)
                    if separator:
                        chunk_text += separator  # 保留分隔符
                    
                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'start_pos': start_pos,
                            'end_pos': start_pos + len(chunk_text)
                        })
                        start_pos += len(chunk_text)
                    
                    # 考虑重叠：保留最后一部分
                    if self.overlap > 0 and len(current_chunk) > 1:
                        # 保留足够的内容用于重叠
                        overlap_text = ''
                        overlap_len = 0
                        for j in range(len(current_chunk) - 1, -1, -1):
                            piece = current_chunk[j]
                            if overlap_len + len(piece) <= self.overlap:
                                overlap_text = piece + (separator if separator else '') + overlap_text
                                overlap_len += len(piece) + (len(separator) if separator else 0)
                            else:
                                break
                        
                        if overlap_text:
                            current_chunk = [overlap_text]
                            current_len = len(overlap_text)
                            start_pos -= len(overlap_text)
                        else:
                            current_chunk = []
                            current_len = 0
                    else:
                        current_chunk = []
                        current_len = 0
                
                # 添加当前 split
                current_chunk.append(split)
                current_len += split_len + (len(separator) if separator and len(current_chunk) > 1 else 0)
        
        # 处理剩余的 chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk) if separator else ''.join(current_chunk)
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(chunk_text)
                })
        
        return chunks
    
    def _force_split(self, text: str, start_pos: int) -> List[Dict]:
        """当没有其他方式时，强制按 chunk_size 切分"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start_pos + i,
                    'end_pos': start_pos + end
                })
            if end >= len(text):
                break
        return chunks
    
    def create_chunks(self, unique_contexts: Dict) -> Tuple[List[Dict], Dict]:
        """
        为所有唯一 context 创建 chunks
        
        Returns:
            (all_chunks, context_to_chunks_map)
        """
        print("\n[Step 2/5] Creating chunks...")
        all_chunks = []
        context_to_chunks = {}
        
        for key, ctx_info in tqdm(unique_contexts.items(), desc="Chunking contexts"):
            chunks = self.chunk_text(ctx_info['full_text'])
            
            # 为每个 chunk 添加 context_id
            start_idx = len(all_chunks)
            for chunk in chunks:
                chunk['context_id'] = ctx_info['id']
                chunk['context_hash'] = key
            
            all_chunks.extend(chunks)
            end_idx = len(all_chunks)
            
            context_to_chunks[ctx_info['id']] = list(range(start_idx, end_idx))
        
        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks, context_to_chunks
    
    @torch.no_grad()
    def get_embeddings(self, texts: List[str], batch_size: int = 32, desc: str = "Embedding") -> torch.Tensor:
        """
        获取文本的向量嵌入
        
        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def embed_chunks(self, chunks: List[Dict]) -> torch.Tensor:
        """嵌入所有 chunks"""
        print("\n[Step 3/5] Embedding chunks...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        return self.get_embeddings(chunk_texts, desc="Embedding chunks")
    
    def embed_questions(self, data: List[Dict]) -> torch.Tensor:
        """嵌入所有问题"""
        print("\n[Step 4/5] Embedding questions...")
        questions = [sample['input'] for sample in data]
        return self.get_embeddings(questions, desc="Embedding questions")
    
    def retrieve_top_k(
        self,
        query_emb: torch.Tensor,
        chunk_embs: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索 Top-K 最相关的 chunks
        
        Returns:
            (indices, scores)
        """
        # 归一化
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        chunk_embs = F.normalize(chunk_embs, p=2, dim=-1)
        
        # 计算余弦相似度
        similarities = torch.matmul(query_emb, chunk_embs.t())
        
        # Top-K
        top_k_scores, top_k_indices = torch.topk(similarities, k, dim=-1)
        
        return top_k_indices, top_k_scores
    
    def convert_dataset(
        self,
        data: List[Dict],
        chunks: List[Dict],
        chunk_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        k: int = 10
    ) -> List[Dict]:
        """
        生成新的 RAG 数据集
        """
        print(f"\n[Step 5/5] Generating RAG dataset with k={k}...")
        new_data = []
        
        for idx, sample in enumerate(tqdm(data, desc="Retrieving chunks")):
            query_emb = question_embeddings[idx].unsqueeze(0)
            
            # 检索 Top-K chunks
            top_k_indices, top_k_scores = self.retrieve_top_k(
                query_emb, chunk_embeddings, k
            )
            
            # 获取检索到的 chunks
            top_k_indices = top_k_indices[0].tolist()
            top_k_scores = top_k_scores[0].tolist()
            
            retrieved_chunks = [chunks[i]['text'] for i in top_k_indices]
            
            # 计算新的总长度
            new_length = sum(len(chunk) for chunk in retrieved_chunks)
            
            # 创建新样本
            new_sample = {
                'input': sample['input'],
                'context': retrieved_chunks,
                'answers': sample['answers'],
                'length': new_length,
                'dataset': 'narrativeqa_rag',
                'language': sample['language'],
                'all_classes': sample['all_classes'],
                '_id': sample['_id'],
                'metadata': {
                    'original_length': sample['length'],
                    'retrieval_k': k,
                    'chunk_scores': top_k_scores,
                    'chunk_indices': top_k_indices
                }
            }
            
            new_data.append(new_sample)
        
        return new_data


def main():
    parser = argparse.ArgumentParser(description='Convert NarrativeQA to RAG format')
    parser.add_argument('--input', type=str, required=True, help='Input narrativeqa.json file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--k', type=int, default=10, help='Number of chunks to retrieve')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size in characters')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap size in characters')
    parser.add_argument('--model', type=str, default='nvidia/NV-Embed-v2', help='Embedding model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # 初始化转换器
    converter = NarrativeQARAGConverter(
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=args.device
    )
    
    # Step 1: 提取唯一 contexts
    unique_contexts = converter.extract_unique_contexts(data)
    
    # Step 2: 创建 chunks
    chunks, context_to_chunks = converter.create_chunks(unique_contexts)
    
    # Step 3: 嵌入 chunks
    chunk_embeddings = converter.embed_chunks(chunks)
    
    # Step 4: 嵌入问题
    question_embeddings = converter.embed_questions(data)
    
    # Step 5: 生成新数据集
    new_data = converter.convert_dataset(
        data, chunks, chunk_embeddings, question_embeddings, k=args.k
    )
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存新数据集
    output_file = output_dir / f'narrativeqa_rag_k{args.k}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Saved RAG dataset to: {output_file}")
    
    # 保存向量
    torch.save(chunk_embeddings, output_dir / 'chunk_embeddings.pt')
    torch.save(question_embeddings, output_dir / 'question_embeddings.pt')
    print(f"Saved embeddings to: {output_dir}")
    
    # 保存 chunk 元数据
    chunk_metadata = {
        'chunks': chunks,
        'unique_contexts': {k: {**v, 'full_text': v['full_text'][:200] + '...'} 
                           for k, v in unique_contexts.items()},  # 截断长文本
        'context_to_chunks': context_to_chunks
    }
    with open(output_dir / 'chunk_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk metadata to: {output_dir / 'chunk_metadata.json'}")
    
    # 保存处理日志
    log = {
        'input_file': args.input,
        'output_dir': str(output_dir),
        'num_samples': len(data),
        'num_unique_contexts': len(unique_contexts),
        'num_chunks': len(chunks),
        'k': args.k,
        'chunk_size': args.chunk_size,
        'overlap': args.overlap,
        'model': args.model
    }
    with open(output_dir / 'processing_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print("Conversion completed successfully!")
    print("="*50)
    print(f"Input samples: {len(data)}")
    print(f"Unique contexts: {len(unique_contexts)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Output file: {output_file}")
    print(f"Average chunks per context: {len(chunks)/len(unique_contexts):.1f}")
    

if __name__ == '__main__':
    main()
