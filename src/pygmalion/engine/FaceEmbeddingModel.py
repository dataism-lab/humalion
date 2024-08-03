from abc import ABC, abstractmethod


class FaceEmbeddingModel(ABC):
    @abstractmethod
    def generate_embeddings(self):
        pass


class InstantID(FaceEmbeddingModel):
    pass
