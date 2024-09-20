from abc import ABC, abstractmethod

from PIL import Image

from src.humalion.engine.persona import ABSPersona


class ImgToPersonaModel(ABC):
    @abstractmethod
    def get_persona(self, image_path: str) -> ABSPersona:
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass
