from sentence_transformers import CrossEncoder

# 🔹 전역 인스턴스: 한 번만 로드되게끔 설정
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_contexts(query: str, candidates: list[str], top_k: int = 2) -> list[str]:
    if not candidates:
        return []

    # query와 candidate를 쌍으로 만들어 예측
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)

    # 점수 기준 정렬
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # 상위 top_k 문장만 반환
    return [text for text, _ in ranked[:top_k]]
