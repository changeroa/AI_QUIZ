import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.embedding import EmbeddingModel
from utils.get_topic import get_topic
from qdrant_client import QdrantClient
from utils.settings import QDRANT_URL, COLLECTION_NAME, STOPWORDS_KO_PATH, STOPWORDS_EN_PATH

# 환경 변수 로드
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Qdrant 및 임베딩 모델 설정
client = QdrantClient(url=QDRANT_URL)
model = EmbeddingModel()

# 퀴즈 1개 생성
def format_prompt(topic: str, context: str, quiz_type: str, config: dict) -> str:
    return f"""
"{topic}"에 대해 아래 문단들을 참고하여 객관식 퀴즈를 만들어줘.
- 보기 4개 포함
- 정답을 명확하게 표기
- 유형: {quiz_type}
- 추가 설정: {config}

문단:
{context}
"""

async def generate_quiz(quiz_type: str, topic: str, config: dict) -> dict:
    query_vec = model.encode(topic)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=3,
        with_payload=True
    )
    context = "\n".join([r.payload["text"] for r in results])

    prompt = format_prompt(topic, context, quiz_type, config)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 교육용 퀴즈 생성기야."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "question": topic,
        "answer": response.choices[0].message.content.strip()
    }

# 여러 개 토픽에 대해 퀴즈 생성
async def generate_quiz_batch(quiz_type: str, config: dict) -> list[dict]:
    topics = get_topic(top_n=5)

    results = []
    for word, _ in topics:
        result = await generate_quiz(quiz_type=quiz_type, topic=word, config=config)
        results.append(result)

    return results
