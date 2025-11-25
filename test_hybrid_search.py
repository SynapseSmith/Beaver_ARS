#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하이브리드 검색 엔진 테스트 스크립트
BM25(40%) + Sentence-BERT(60%) 검색 기능 검증
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 테스트 메뉴 데이터
menu_info = {
    "짜장면": "6500",
    "짬뽕": "7800",
    "탕수육": "18000",
    "볶음밥": "7000",
    "군만두": "6000",
    "고기짬뽕": "9500",
    "해물짬뽕": "9000",
    "삼선짬뽕": "10000",
}

def initialize_hybrid_search(info_dict):
    """하이브리드 검색 초기화"""
    keys = list(info_dict.keys())
    
    # 1. Sentence-BERT 임베딩
    print("📊 Sentence-BERT 모델 로딩...")
    embedding_model = SentenceTransformer('nlpai-lab/KoE5')
    embeddings = embedding_model.encode(keys)
    
    # 2. BM25 초기화
    print("🔍 BM25 인덱스 생성...")
    tokenized_corpus = [list(key) for key in keys]  # 한글 글자 단위 토큰화
    bm25 = BM25Okapi(tokenized_corpus)
    
    return keys, embeddings, bm25, embedding_model

def hybrid_search(query, info_dict, keys, embeddings, bm25, embedding_model):
    """하이브리드 검색 실행"""
    print(f"\n{'='*60}")
    print(f"🔎 검색어: '{query}'")
    print(f"{'='*60}")
    
    # 1. Sentence-BERT 유사도
    query_embedding = embedding_model.encode([query])[0]
    sbert_scores = util.cos_sim(query_embedding, embeddings).squeeze().tolist()
    sbert_scores = np.array(sbert_scores)
    
    # 2. BM25 유사도
    tokenized_query = list(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 3. 정규화
    sbert_normalized = sbert_scores
    bm25_normalized = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
    
    # 4. 하이브리드 스코어
    hybrid_scores = 0.4 * bm25_normalized + 0.6 * sbert_normalized
    
    # 5. Top 3 결과 출력
    top_indices = np.argsort(hybrid_scores)[::-1][:3]
    
    print("\n📊 검색 결과 (Top 3):")
    print(f"{'순위':<5} {'메뉴':<15} {'가격':<10} {'BM25':<10} {'SBERT':<10} {'Hybrid':<10}")
    print("-" * 70)
    
    for rank, idx in enumerate(top_indices, 1):
        menu = keys[idx]
        price = info_dict[menu]
        bm25_score = bm25_normalized[idx]
        sbert_score = sbert_normalized[idx]
        hybrid_score = hybrid_scores[idx]
        
        print(f"{rank:<5} {menu:<15} {price:<10} {bm25_score:>7.3f}   {sbert_score:>7.3f}   {hybrid_score:>7.3f}")
    
    # 최고 매칭
    best_idx = top_indices[0]
    best_menu = keys[best_idx]
    best_price = info_dict[best_menu]
    
    print(f"\n✅ 최종 선택: {best_menu} ({best_price}원)")
    print(f"   - BM25 기여도: {0.4 * bm25_normalized[best_idx]:.3f}")
    print(f"   - SBERT 기여도: {0.6 * sbert_normalized[best_idx]:.3f}")
    print(f"   - 최종 스코어: {hybrid_scores[best_idx]:.3f}")
    
    return best_menu, best_price

def main():
    print("🦫 Beaver ARS 하이브리드 검색 엔진 테스트")
    print("=" * 60)
    
    # 초기화
    keys, embeddings, bm25, embedding_model = initialize_hybrid_search(menu_info)
    
    # 테스트 쿼리
    test_queries = [
        "짜장면 가격 알려줘",
        "짬뽕 얼마야?",
        "고기 들어간 짬뽕",
        "해물짬뽕 주세요",
        "탕수육 시키고 싶어요",
        "짜장 하나요",
    ]
    
    for query in test_queries:
        hybrid_search(query, menu_info, keys, embeddings, bm25, embedding_model)
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("\n💡 하이브리드 검색 장점:")
    print("  • BM25: 키워드 정확 매칭 (예: '짜장' → '짜장면')")
    print("  • SBERT: 의미 이해 (예: '얼마야' → 가격 정보)")
    print("  • 결합 효과: 키워드 + 의미를 모두 고려한 최적 매칭")

if __name__ == "__main__":
    main()
