import numpy as np

from src.humalion.engine.face_detection_model import FaceDetectionModel
from src.humalion.engine.face_embedding_model import FaceEmbeddingModel
from src.humalion.engine.face_swapper import FaceSwapperModel
from src.humalion.engine.image_generative_model import ImageGenerativeModel
from src.humalion.engine.persona import ABSPersona
from src.humalion.engine.user_photo import UserPhoto
from IPython.display import Image, display
from abc import ABC, abstractmethod


class ABCDHuman(ABC):
    def __init__(self, voice=None, face_embeddings=None, persona=None):
        self.__voice = voice
        self.__face_embeddings = face_embeddings
        self.__persona = persona

    @property
    def persona(self):
        return self.__persona

    @persona.setter
    def persona(self, value):
        self.__persona = value

    @property
    def face_embeddings(self):
        return self.__face_embeddings

    @face_embeddings.setter
    def face_embeddings(self, value):
        self.__face_embeddings = value

    @property
    def voice(self):
        return self.__voice

    @voice.setter
    def voice(self, value):
        self.__voice = value

    @abstractmethod
    def generate_photo(self, prompt: str) -> str:
        """
        Args:
            prompt:

        Returns:

        """
        pass

    @abstractmethod
    def show(self):
        pass


class TwinDHuman(ABCDHuman):
    def __init__(self, user_photos: list[UserPhoto], voice=None, face_embeddings=None, persona=None):
        self.user_photos = user_photos
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)


class SyntheticDHuman(ABCDHuman):
    def __init__(
            self,
            persona: ABSPersona,
            generative_model: ImageGenerativeModel | None = None,
            face_swap: bool = True,
            face_embedding_model: FaceEmbeddingModel | None = None,
            face_swap_model: FaceSwapperModel | None = None,
            voice=None,
            face_embeddings: np.ndarray | None = None,
    ):
        if face_swap and (face_embedding_model is None or face_swap_model is None):
            raise ValueError("If face_swap is True, face_embedding_model and face_swap_model must be set")

        self.generative_model = generative_model
        self.face_swap = face_swap
        self.face_embedding_model = face_embedding_model
        self.face_swap_model = face_swap_model

        self._prepare_models()
        self.source_photo_path = self.generative_model.generate_photo(persona.prompt())
        self.face_embeddings = face_embeddings if face_embeddings else self.face_embedding_model.generate_embeddings([self.source_photo_path])[0]

        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)

    def _prepare_models(self):
        self.generative_model.prepare_model()

        if self.face_embedding_model:
            self.face_embedding_model.prepare_model()

        if self.face_swap_model:
            self.face_swap_model.prepare_model()

    def _get_source_photo(self):
        return Image(self.source_photo_path)

    def show(self):
        display(self._get_source_photo())
        print(self.persona.prompt())

    def generate_photo(self, prompt: str) -> str:
        # source_img_path = self.generative_model.generate_photo(prompt=prompt)
        result_img_path = self.face_swap_model.swap_face(
            source_image_path=self.source_photo_path,
            face_emb=self.face_embeddings,
            prompt=prompt,
            gender=self.persona.gender,  # todo поправить на менее зависимое
        )
        return result_img_path

