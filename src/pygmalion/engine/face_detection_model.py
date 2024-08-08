from abc import ABC, abstractmethod


class FaceDetectionModel(ABC):
    @abstractmethod
    def detect(self):
        pass