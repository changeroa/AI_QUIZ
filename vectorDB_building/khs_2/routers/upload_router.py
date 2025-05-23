#upload_router.py

from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from qdrant_client.http.models import PointStruct
from shared import qdrant_client, embedding_model, COLLECTION_NAME
import uuid, json

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/text", summary="단일 문장 벡터 저장")
def add_text(text: str = Query(..., description="저장할 문장")):
    embedding = embedding_model.encode(text)
    point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": text})
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"status": "저장 완료", "text": text}


@router.post("/json", summary="JSON 파일 업로드", description="JSON 파일을 업로드하여 여러 문장을 벡터화 후 저장")
async def upload_json(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="JSON 파일만 업로드 가능합니다.")

    content = await file.read()
    try:
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {e}")

    points = []
    for item in data:
        embedding = embedding_model.encode(item["text"])
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "id": item.get("id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "text": item.get("text")
            }
        )
        points.append(point)

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "업로드 및 저장 완료", "count": len(points)}
