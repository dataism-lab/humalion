from typing import Any


class FaceEmbedding:
    def __init__(self, source_image: list[Any], embedding=None):
        self.embedding = embedding
        self.source_image = source_image


