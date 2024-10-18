from abc import ABC, abstractmethod

import numpy as np

from ..utils.enum import StrEnum


class FaceSwapperModel(ABC):
    class Gender(StrEnum):
        MALE = "male"
        FEMALE = "female"

    @abstractmethod
    def swap_face(self, source_image_path: str, face_emb: np.ndarray, prompt: str, gender: Gender) -> str:
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass
