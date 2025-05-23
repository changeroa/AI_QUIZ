from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Dict


# 🔹 퀴즈 유형 Enum: Swagger에서 드롭다운으로 선택 가능
class QuizType(str, Enum):
    mcq = "mcq"
    ox = "ox"
    short = "short"


# 🔹 객관식 보기 항목 구조
class QuizOption(BaseModel):
    option_text: str = Field(..., description="보기 텍스트")
    is_correct: bool = Field(..., description="정답 여부 (true: 정답)")

    
# 🔹 단일 퀴즈 응답 모델
class QuizResponse(BaseModel):
    question: str = Field(..., description="생성된 문제")
    options: Optional[List[QuizOption]] = Field(None, description="보기 목록 (객관식, OX 문제일 경우)")
    answer: str = Field(..., description="정답")


# 🔹 퀴즈 다중 응답 모델
class QuizBatchResponse(BaseModel):
    quizzes: List[QuizResponse] = Field(..., description="생성된 퀴즈 목록")


# 🔹 단일 퀴즈 생성 요청

class QuizRequest(BaseModel):
    quiz_type: QuizType = Field(
        default=QuizType.mcq,
        description="퀴즈 유형: mcq / ox / short"
    )
    topic: str = Field(
        default="git pull",
        description="문제의 주제 또는 키워드"
    )
    config: Dict = Field(
        default={
            "difficulty": "medium",
            "bloom_level": "application",
            "distractor_count": 3
        },
        description="퀴즈 생성 설정 (유형에 따라 다름)"
    )



# 🔹 다중 퀴즈 생성 요청 (자동 토픽)
class QuizBatchRequest(BaseModel):
    quiz_type: QuizType = Field(..., description="퀴즈 유형: mcq / ox / short")
    config: Dict = Field(
        ..., 
        example={
            "difficulty": "low"
        },
        description="퀴즈 생성 설정"
    )


# 🔹 객관식 퀴즈 구성 설정
class MCQPromptConfig(BaseModel):
    difficulty: str = Field(default="medium", description="퀴즈 난이도")
    bloom_level: str = Field(default="application", description="Bloom's Taxonomy 단계")
    distractor_count: int = Field(default=3, description="오답 보기 개수")


# 🔹 OX 퀴즈 구성 설정
class OXPromptConfig(BaseModel):
    difficulty: str = Field(default="medium", description="퀴즈 난이도")


# 🔹 단답형 퀴즈 구성 설정
class ShortPromptConfig(BaseModel):
    difficulty: str = Field(default="medium", description="퀴즈 난이도")
    word_limit: int = Field(default=20, description="정답 최대 단어 수")
