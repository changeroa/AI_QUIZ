#vector_router.py
from fastapi import APIRouter, Query
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter
from utils.get_topic import get_topic
from shared import qdrant_client, embedding_model, COLLECTION_NAME

router = APIRouter()


@router.get("/search", tags=["Vector"], summary="벡터 검색", description="입력 문장과 유사한 문서 검색 (옵션: 임계값)")
def search_text(
    query: str = Query(..., description="검색할 문장"),
    limit: int = Query(3, description="최대 검색 결과 수"),
    threshold: float = Query(None, description="점수 임계값 (선택)")
):
    query_vec = embedding_model.encode(query)

    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "query_vector": query_vec,
        "limit": limit,
        "with_payload": True
    }
    if threshold is not None:
        search_kwargs["score_threshold"] = threshold

    results = qdrant_client.search(**search_kwargs)

    return {
        "matches": [
            {"text": r.payload.get("text", ""), "score": r.score}
            for r in results
        ]
    }


@router.get("/all", tags=["Vector"], summary="전체 출력", description="DB내 모든 데이터 출력.")
def get_all():
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_vectors=True
    )
    return [
        {
            "id": p.id,
            "text": p.payload.get("text", ""),
            "vector": p.vector[:5]  # 벡터 앞 5차원만 예시 출력
        }
        for p in points
    ]


@router.get("/vector/top-words", tags=["Vector"], summary="최빈값", description="최빈 단어 10개 출력.")
def get_top_words_from_qdrant():
    result = get_topic(top_n=10)
    return {"top_words": result}


@router.delete("/delete", tags=["Vector"], summary="전체 삭제", description="DB내 모든 데이터 삭제.")
def delete_all():
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[])
    )
    return {"status": "전체 데이터 삭제 완료"}
