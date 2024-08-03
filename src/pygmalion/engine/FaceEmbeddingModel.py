from abc import ABC, abstractmethod


class FaceEmbeddingModel(ABC):
    @abstractmethod
    def generate_embeddings(self, source_image):
        pass


class InstantID(FaceEmbeddingModel):
    pass
