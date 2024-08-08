from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    @abstractmethod
    def generate_photo(self):
        pass
