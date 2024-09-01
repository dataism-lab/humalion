from humalion.engine.face_detection_model import FaceDetectionModel
from humalion.engine.face_embedding_model import FaceEmbeddingModel
from humalion.engine.generative_model import GenerativeModel
from humalion.engine.persona import ABSPersona
from humalion.engine.user_photo import UserPhoto


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
        self.user_photos = user_photos
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)


class SyntheticDHuman(ABCDHuman):
    def __init__(
            self,
            persona: ABSPersona,
            generative_model: GenerativeModel | None = None,
            face_swap: bool = True,
            face_embedding_model: FaceEmbeddingModel | None = None,
            voice=None,
            face_embeddings=None,
    ):
        if face_swap and face_embedding_model is None:
            raise ValueError("If face_swap is True, face_embedding_model must be set")

        self.generative_model = generative_model
        self.face_swap = face_swap
        self.face_embedding_model = face_embedding_model

        self._prepare_models()
        self.source_photo_path = self.generative_model.generate_photo(persona.prompt())
        self.face_embeddings = self.face_embedding_model.generate_embeddings(self.source_photo_path)

        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)

    def _prepare_models(self):
        self.generative_model.prepare_model()

        if self.face_swap:
            self.face_embedding_model.prepare_model()
