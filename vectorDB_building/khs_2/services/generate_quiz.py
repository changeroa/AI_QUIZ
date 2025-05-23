import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
from pydantic import ValidationError

from shared import qdrant_client, embedding_model, COLLECTION_NAME
from utils.get_topic import get_topic
from prompts import generate_mcq_prompt, generate_ox_prompt, generate_short_prompt
from schemas import MCQPromptConfig, OXPromptConfig, ShortPromptConfig, QuizOption, QuizResponse, QuizType

# 환경 변수 로드
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 퀴즈 1개 생성
async def generate_quiz(quiz_type: QuizType, topic: str, config: dict) -> QuizResponse:
    if quiz_type == "mcq":
        prompt = generate_mcq_prompt(topic, MCQPromptConfig(**config))
    elif quiz_type == "ox":
        prompt = generate_ox_prompt(topic, OXPromptConfig(**config))
    elif quiz_type == "short":
        prompt = generate_short_prompt(topic, ShortPromptConfig(**config))
    else:
        raise ValueError(f"Invalid quiz_type: {quiz_type}")

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates quiz questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=350,
        temperature=0.3
    )
    content = response.choices[0].message.content

    try:
        parsed = validate_json(content)
    except Exception as e:
        raise ValueError(f"GPT 응답 오류: {e}")

    # 보기 구성
    options = []
    if quiz_type == "ox":
        options = [
            QuizOption(option_text="O", is_correct=parsed["correct_answer"] == "O"),
            QuizOption(option_text="X", is_correct=parsed["correct_answer"] == "X"),
        ]
    elif quiz_type == "short":
        options = []
    else:  # mcq
        correct = parsed["correct_answer"]
        for key in ["A", "B", "C", "D"]:
            options.append(
                QuizOption(option_text=parsed[key], is_correct=(key == correct))
            )

    return QuizResponse(
        question=parsed["question"],
        options=options if options else None,
        answer=parsed["correct_answer"]
    )

# JSON 응답 유효성 검사
def validate_json(content: str) -> dict:
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found.")
        json_str = match.group()
        return json.loads(json_str)
    except Exception as e:
        raise ValidationError(f"Invalid JSON format: {e}")

# 다중 퀴즈 생성
async def generate_quiz_batch(quiz_type: str, config: dict) -> List[QuizResponse]:
    topics = get_topic(top_n=5)
    results = []
    for word, _ in topics:
        result = await generate_quiz(quiz_type=quiz_type, topic=word, config=config)
        results.append(result)
    return results
