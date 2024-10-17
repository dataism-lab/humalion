import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from pydantic import BaseModel, FilePath, ConfigDict

from src.humalion.engine.errors import ModelInferenceError
from src.humalion.engine.face_embedding import PersonEmbedding
from src.humalion.engine.face_embedding_model import FaceEmbeddingModel

from src.humalion.engine.img_to_persona_model import ImgToPersonaModel
from src.humalion.engine.persona import ABSPersona
from src.humalion.models.instant_id.face_analysis import FixedFaceAnalysis
from src.humalion.models.instant_id.utils import resize_img
from src.humalion.utils.mixins import CheckOrCreateDirMixin


class PersonData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: FilePath
    face_data: dict
    appearance_data: ABSPersona | None = None


class InstantIDFaceEmbModel(FaceEmbeddingModel, CheckOrCreateDirMixin):
    BASE_RESOURCE_DIR = Path('humalion_resources')

    def __init__(
            self,
            img_to_pers_model: ImgToPersonaModel,
            max_threads: int = 8,
            output_dir: str = 'output/instant_id_emb/'
    ):
        self.img_to_pers_model = img_to_pers_model
        self.face_info_model = None
        self.max_threads = max_threads
        self.output_dir = output_dir
        self._check_dir(self.output_dir)

    def _init_face_analysis(self) -> None:
        self.face_info_model = FixedFaceAnalysis(
            name="antelopev2", root=self.BASE_RESOURCE_DIR, providers=['CUDAExecutionProvider', "CPUExecutionProvider"]
        )
        self.face_info_model.prepare(ctx_id=0, det_size=(320, 320))

    def prepare_model(self):
        self._init_face_analysis()

    def _get_face_info(self, image: Image) -> dict | None:
        """
        Analyze face on image using insightface
        Args:
            image: PIL image for face analyze

        Returns: dict with face info
            dict[
                'bbox': np.array,
                'kps': np.array,
                'det_score': float,
                'landmark_3d_68': np.array,
                'pose': np.array,
                'landmark_2d_106': np.array,
                'gender': int,
                'age': int,
                'embedding': np.array,
            ]
        """
        face_image: Image = resize_img(image)

        face_info_list: list[Face] = self.face_info_model.get(
            cv2.cvtColor(
                np.array(face_image),
                cv2.COLOR_RGB2BGR,
            ),
        )

        face_info: dict | None = None
        if len(face_info_list) > 0:
            face_info = sorted(
                face_info_list,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            )[-1]  # only use the maximum face

        return face_info

    def _get_appearance_info(self, image_path: str) -> ABSPersona:
        return self.img_to_pers_model.get_persona(image_path=image_path)

    def _get_person_data(self, image_path: str) -> PersonData | None:
        image = Image.open(image_path)
        face_data = self._get_face_info(image)
        if face_data:
            appearance_data = self._get_appearance_info(image_path)
            return PersonData(image_path=image_path, face_data=face_data, appearance_data=appearance_data)

        return None

    def generate_embeddings(self, source_photo_paths: list[str]) -> tuple[np.ndarray, str]:
        """
        Args:
            source_photo_paths: A list of file paths to source photos for which embeddings need to be generated.

        Returns:
            A tuple containing the face embeddings as a NumPy array of dtype float32 and the appearance embeddings prompt string.
        """
        with ThreadPoolExecutor(max_workers=min(len(source_photo_paths), self.max_threads)) as executor:
            person_data_list = list(executor.map(self._get_person_data, source_photo_paths))

        if not source_photo_paths:
            raise ModelInferenceError(f'Embeddings of images: {source_photo_paths} have not been generated.')
        elif len(source_photo_paths) > 1:
            face_embeddings: list[np.array] = []
            appearance_embeddings: list[ABSPersona] = []

            for person_data in person_data_list:
                face_embeddings.append(person_data.face_data['embedding'])
                if person_data.appearance_data:
                    appearance_embeddings.append(person_data.appearance_data)

            if face_embeddings:
                result_face_emb = np.mean(face_embeddings, axis=0)
            if appearance_embeddings:
                persona_class = appearance_embeddings[0].__class__
                result_appearance_emb = persona_class.mean(appearance_embeddings)
        elif len(source_photo_paths) == 1:
            result_face_emb = person_data_list[0].face_data['embedding']
            result_appearance_emb = person_data_list[0].appearance_data
        else:
            raise ModelInferenceError(f'Embeddings of images: {source_photo_paths} have not been generated.')

        return np.array(result_face_emb, dtype=np.float32), np.array(result_appearance_emb.prompt())  # noqa

    def save_embeddings(self, embeddings: Iterable[np.ndarray]) -> str:
        file_name = f"{uuid.uuid4()}.npz"
        file_path = Path(self.output_dir) / file_name
        np.savez(file_path, *embeddings)
        return file_name
