from fastapi import APIRouter, HTTPException
from services.generate_quiz import generate_quiz, generate_quiz_batch
from schemas.quiz_schemas import QuizRequest

router = APIRouter()

@router.post("/generate-batch", summary="문제생성", description="최빈 단어 5개로 문제 생성")
async def create_quiz_batch(request: QuizRequest):
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
