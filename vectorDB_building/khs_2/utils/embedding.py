
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    #임베딩 모델 별 차원 추출가능하게 편집(env에서 차원 제거)