import random
from os import PathLike
from pathlib import Path
from typing import Iterable

from ..engine.image_generative_model import ImageGenerativeModel


class RandomLocalPhotoGenerativeModel(ImageGenerativeModel):
    DEFAULT_ALLOWED_EXTENSIONS = frozenset((".png", ".jpg", ".jpeg"))

    def __init__(self, photos_path: str | PathLike, allowed_extensions: Iterable[str] = DEFAULT_ALLOWED_EXTENSIONS):
        super().__init__()
        self.photos_path = photos_path
        self.allowed_extensions = allowed_extensions
        self.photos_list = self._get_list_of_photos()

    def _get_list_of_photos(self) -> list[str]:
        return [str(p) for p in Path(self.photos_path).glob("**/*") if p.suffix in self.allowed_extensions]

    def generate_photo(self, prompt: str) -> str:
        return random.choice(self.photos_list)

    def prepare_model(self):
        pass
