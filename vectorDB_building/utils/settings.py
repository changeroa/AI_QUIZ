import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "korean-texts")
STOPWORDS_KO_PATH = os.getenv("STOPWORDS_KO_PATH", "utils/stopwords-ko.txt")
STOPWORDS_EN_PATH = os.getenv("STOPWORDS_EN_PATH", "utils/stopwords-en.txt")
