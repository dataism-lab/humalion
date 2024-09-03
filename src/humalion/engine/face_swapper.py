from abc import ABC, abstractmethod


class FaceSwapperModel(ABC):
    @abstractmethod
    def swap_face(self, source_image) -> str:
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass


