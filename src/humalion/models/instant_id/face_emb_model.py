from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from pydantic import BaseModel, FilePath, ConfigDict

from src.humalion.engine.face_embedding import PersonEmbedding
from src.humalion.engine.face_embedding_model import FaceEmbeddingModel

from src.humalion.engine.img_to_persona_model import ImgToPersonaModel
from src.humalion.engine.persona import ABSPersona
from src.humalion.models.instant_id.face_analysis import FixedFaceAnalysis
from src.humalion.models.instant_id.utils import resize_img


class PersonData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: FilePath
    face_data: dict
    appearance_data: ABSPersona | None = None


class InstantIDFaceEmbModel(FaceEmbeddingModel):
    BASE_RESOURCE_DIR = Path('humalion_resources')
    def __init__(
            self,
            img_to_pers_model: ImgToPersonaModel
    ):
        self.img_to_pers_model = img_to_pers_model
        self.face_embeddings: list[np.array] = []
        self.appearance_embeddings: list[dict] = []
        self.result_embeddings: list[PersonEmbedding] = []
        self.face_info_model = None

    def _init_face_analysis(self) -> None:
        self.face_info_model = FixedFaceAnalysis(
            name="antelopev2", root=self.BASE_RESOURCE_DIR, providers=['CUDAExecutionProvider', "CPUExecutionProvider"]
        )
        self.face_info_model.prepare(ctx_id=0, det_size=(320, 320))

    def prepare_model(self):
        self._init_face_analysis()

    def _get_face_info(self, image: Image) -> dict | None:
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

    def _get_appearance_info(self, image_path: str):
        return self.img_to_pers_model.get_persona(image_path=image_path)

    def _get_person_data(self, image_path: str) -> PersonData | None:
        image = Image.open(image_path)
        face_data = self._get_face_info(image)
        if face_data:
            app_data = self._get_appearance_info(image_path)
            return PersonData(image_path=image_path, face_data=face_data, appearance_data=app_data)

        return None

    def generate_embeddings(self, source_photo_paths: list[str]):
        with ThreadPoolExecutor(max_workers=len(source_photo_paths)) as executor:
             person_data_list = list(executor.map(self._get_person_data, source_photo_paths))


#
# def make_avatar(
#         out_fpath: str,
#         image_path_list: list[str],
#         gender_profile: Gender | None,
#         stylename=NO_STYLENAME,
# ) -> dict[str, str | list[str]]:
#     global _face_info_model
#     global _instaid_pipe
#     global _poses_women
#     global _poses_men
#
#     prompt = ""
#     face_embeddings: list[np.array] = []
#     appearance_embeddings: list[dict] = []
#     gender_list: list[int] = []
#
#     gender = gender_profile
#     face_emb: np.array | None = None
#
#     st_all = time.time()
#     used_img_paths: list[str] = []
#     global SHARED_THREAD_POOL_EXECUTOR
#     tasks = [SHARED_THREAD_POOL_EXECUTOR.submit(get_face_and_appearance_info, image_path=image_path) for image_path in
#              image_path_list]
#     for image_path, task in zip(image_path_list, tasks):
#         _, ref_info, dt = task.result()
#         if ref_info is not None:
#             face_embeddings.append(ref_info["embedding"])
#             gender_list.append(ref_info["gender"])
#             used_img_paths.append(image_path)
#             appearance_embeddings.append(dt)
#     logger.info(f"All preprocessing time = {time.time() - st_all} sec")
#
#     Path(out_fpath).parent.mkdir(exist_ok=True, parents=True)
#     if len(face_embeddings) > 0:
#         face_emb = np.mean(face_embeddings, axis=0)
#         logger.info(f"Appearance embeddings: {appearance_embeddings}")
#         appearance_emb = appearance_mean(appearance_embeddings)
#         appearance_emb_text = get_prompt_post(appearance_emb)
#
#         # Save embs
#         out_embs_fpath: str = (Path(out_fpath).parent / FACEEMB_FNAME).as_posix()
#         emb_ndarray = np.array(face_embeddings, dtype=np.float32)
#         logger.info(f"Embeddings shape: {emb_ndarray.shape}, saved to {out_embs_fpath}")
#
#         data = [emb_ndarray, np.array(appearance_emb_text)]
#         np.savez(out_embs_fpath, *data)
#     else:
#         raise PygmaAPIError("Embeddings were not computed!")
#
#     if gender is None and len(gender_list) > 0:
#         gender_val = round(sum(gender_list) / len(gender_list))
#         if gender_val == 0:
#             gender = Gender.female
#         else:
#             gender = Gender.male
#     elif gender is None:
#         raise PygmaAPIError("Gender were not computed!")
#
#     black = Image.new("RGB", AVA_SIZE, (0, 0, 0))
#     if gender == Gender.female:
#         prompt = "portrait photo of a dressed in high quality clothes woman," + " " + get_prompt_avatar(appearance_emb)
#         idx = random.randint(0, _poses_women.shape[0] - 1)  # type: ignore
#         logger.info(f"Pose idx: {idx}")
#         kps = _poses_women[idx]  # type: ignore
#         face_kps = draw_kps(black, kps)  # - use random kps from women kps file
#     else:
#         prompt = "portrait photo of a dressed in high quality clothes man," + " " + get_prompt_avatar(appearance_emb)
#         idx = random.randint(0, _poses_men.shape[0] - 1)  # type: ignore
#         logger.info(f"Pose idx: {idx}")
#         kps = _poses_men[idx]  # type: ignore
#         face_kps = draw_kps(black, kps)  # - use random kps from men kps file
#
#     prompt = prompt + GENERAL_POSITIVE_PROMPT_POSTFIX
#
#     logger.info(f"Generating avatar with prompt {prompt}, negative prompt {GENERAL_NEGATIVE_PROMPT_POSTFIX}")
#
#     if not IS_DEV_VERSION:
#         with CALL_INSTANTID_LOCKER:
#             _instaid_pipe.set_ip_adapter_scale(0.8)
#             image = _instaid_pipe(
#                 prompt=prompt,
#                 negative_prompt=GENERAL_NEGATIVE_PROMPT_POSTFIX,
#                 image_embeds=face_emb,
#                 image=face_kps,
#                 controlnet_conditioning_scale=0.5,
#                 num_inference_steps=NUM_STEPS,
#                 guidance_scale=3.0,
#             ).images[0]
#
#         Path(out_fpath).parent.mkdir(exist_ok=True, parents=True)
#         save_image(image, out_fpath)
#         logger.info(f"Generated {out_fpath}")
#     else:
#         shutil.copyfile(random.choice(used_img_paths), out_fpath)
#
#     return {
#         "local_paths": [out_fpath] + used_img_paths,
#         "gender": gender.value,  # type: ignore
#         "artifact_paths": [out_embs_fpath],
#     }