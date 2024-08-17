from pygmalion.engine.face_detection_model import FaceDetectionModel
from pygmalion.engine.face_embedding_model import FaceEmbeddingModel
from pygmalion.engine.generative_model import GenerativeModel
from pygmalion.engine.persona import ABSPersona
from pygmalion.engine.user_photo import UserPhoto


class ABCDHuman:
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


class TwinDHuman(ABCDHuman):
    def __init__(self, user_photos: list[UserPhoto], voice=None, face_embeddings=None, persona=None):
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)
        self.user_photos = user_photos


class SyntheticDHuman(ABCDHuman):
    def __init__(
            self,
            persona: ABSPersona,
            generative_model: GenerativeModel | None = None,
            face_swap: bool = True,
            face_detection_model: FaceDetectionModel | None = None,
            face_embedding_model: FaceEmbeddingModel | None = None,
            voice=None,
            face_embeddings=None,
    ):
        self.generative_model = generative_model
        self.face_swap = face_swap
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)

