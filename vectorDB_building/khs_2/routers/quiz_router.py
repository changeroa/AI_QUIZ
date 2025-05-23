#quiz_router.py

from fastapi import APIRouter, HTTPException
from services.generate_quiz import generate_quiz, generate_quiz_batch
from schemas.quiz_schemas import QuizRequest, QuizBatchRequest, QuizResponse, QuizBatchResponse

router = APIRouter()

@router.post(
    "/generate",
    tags=["Quiz"],
    summary="단일 문제 생성",
    description="지정한 topic에 대해 1개의 퀴즈 생성",
    response_model=QuizResponse
)
async def create_single_quiz(request: QuizRequest):
    return await generate_quiz(
        quiz_type=request.quiz_type,
        topic=request.topic,
        config=request.config
    )


@router.post(
    "/generate-batch",
    tags=["Quiz"],
    summary="최빈 단어 기반 문제 일괄 생성",
    description="상위 5개 topic 기준 퀴즈 자동 생성",
    response_model=QuizBatchResponse
)
async def create_quiz_batch(request: QuizBatchRequest):
    try:
        result = await generate_quiz_batch(
            quiz_type=request.quiz_type,
            config=request.config
        )
        return {"quizzes": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
