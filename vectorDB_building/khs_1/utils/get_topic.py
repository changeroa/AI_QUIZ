import re
from collections import Counter
from qdrant_client import QdrantClient
from utils.settings import QDRANT_URL, COLLECTION_NAME, STOPWORDS_KO_PATH, STOPWORDS_EN_PATH

def load_stopwords(filepath: str) -> set:
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def clean_text(text: str) -> str:
    return re.sub(r'[^가-힣a-zA-Z\s]', '', text)

def tokenize_and_clean(text: str, stopwords_ko: set, stopwords_en: set) -> list[str]:
    text = clean_text(text)
    tokens = text.split()
    return [
        token for token in tokens
        if len(token) > 1 and token.lower() not in stopwords_en and token not in stopwords_ko
    ]

def get_topic(top_n: int = 10):
    # Qdrant 연결
    client = QdrantClient(url=QDRANT_URL)
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10_000, with_payload=True)

    # 모든 텍스트 수집
    texts = [p.payload.get("text", "") for p in points if "text" in p.payload]
    all_text = " ".join(texts)

    # 불용어 로드
    stopwords_ko = load_stopwords(STOPWORDS_KO_PATH)
    stopwords_en = load_stopwords(STOPWORDS_EN_PATH)

    # 토큰화 및 카운팅
    tokens = tokenize_and_clean(all_text, stopwords_ko, stopwords_en)
    freq = Counter(tokens)
    return freq.most_common(top_n)
