
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def get_model_name(self) -> str:
        return self.model_name
