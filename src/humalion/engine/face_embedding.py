from dataclasses import dataclass

import numpy as np


@dataclass
class PersonEmbedding:
    face_embedding: np.ndarray
    appearance_embedding: np.ndarray
