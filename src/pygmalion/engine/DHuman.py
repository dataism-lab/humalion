from pygmalion.engine.UserPhoto import UserPhoto


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
    def __init__(self,  user_photos: list[UserPhoto], voice=None, face_embeddings=None, persona=None):
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)
        self.user_photo = user_photos

        
class SyntheticDHuman(ABCDHuman):
    def __init__(self, generated_photo, voice=None, face_embeddings=None, persona=None):
        super().__init__(voice=voice, face_embeddings=face_embeddings, persona=persona)
        self.generated_photo = generated_photo
