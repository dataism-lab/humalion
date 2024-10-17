from abc import ABC, abstractmethod
from typing import Any


class FaceEmbeddingModel(ABC):
    @abstractmethod
    def generate_embeddings(self, source_images: list[str]):
        pass

    @abstractmethod
    def save_embeddings(self, embeddings: Any) -> str:
        """
        must return path to saved embeddings file
        """
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass


