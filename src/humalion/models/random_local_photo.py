from os import PathLike
from pathlib import Path
import random

from src.humalion.engine.generative_model import GenerativeModel


class RandomLocalPhotoGenerativeModel(GenerativeModel):
    def __init__(self, photos_path: str | PathLike):
        super().__init__()
        self.photos_path = photos_path
        self.photos_list = self._get_list_of_photos()

    def _get_list_of_photos(self) -> list[str]:
        path = Path(self.photos_path).glob('**/*.{png,gif,jpg,jpeg}')
        files = [str(x) for x in path if x.is_file()]
        return files

    def generate_photo(self, prompt: str) -> str:
        return random.choice(self.photos_list)

    def prepare_model(self):
        pass

