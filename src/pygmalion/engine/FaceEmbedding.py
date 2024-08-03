from typing import Any


class FaceEmbedding:
    def __init__(self, source_images: list[Any], embeddings=None):
        self.embeddings = embeddings
        self.source_images = source_images


