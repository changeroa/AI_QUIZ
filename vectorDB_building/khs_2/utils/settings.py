import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Qdrant 설정
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "korean-texts")

# 모델 설정
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
)

# 불용어 파일 경로
STOPWORDS_KO_PATH = os.getenv("STOPWORDS_KO_PATH", "utils/stopwords-ko.txt")
STOPWORDS_EN_PATH = os.getenv("STOPWORDS_EN_PATH", "utils/stopwords-en.txt")

# 필수 항목 누락 시 예외 발생
assert QDRANT_URL is not None, "QDRANT_URL is not set in .env"
assert COLLECTION_NAME is not None, "COLLECTION_NAME is not set in .env"
assert EMBEDDING_MODEL_NAME is not None, "EMBEDDING_MODEL_NAME is not set in .env"
