#admin_router.py

from fastapi import APIRouter
from qdrant_client.http.models import VectorParams, Distance
from shared import qdrant_client, COLLECTION_NAME, VECTOR_DIM

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.post("/init", summary="컬렉션 초기화", description="Qdrant 컬렉션을 삭제 후 재생성합니다.")
def init_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    return {"status": "컬렉션 초기화 완료"}
