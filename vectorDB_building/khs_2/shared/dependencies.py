from qdrant_client import QdrantClient
from utils.embedding import EmbeddingModel
from utils.settings import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL_NAME

embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
VECTOR_DIM = embedding_model.get_dimension()
qdrant_client = QdrantClient(url=QDRANT_URL)
