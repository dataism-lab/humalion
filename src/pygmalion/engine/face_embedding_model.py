from abc import ABC, abstractmethod


class FaceEmbeddingModel(ABC):
    @abstractmethod
    def generate_embeddings(self, source_image):
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass


