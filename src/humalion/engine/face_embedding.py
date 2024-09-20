from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class PersonEmbedding:
    face_embedding: np.ndarray
    appearance_embedding: np.ndarray



