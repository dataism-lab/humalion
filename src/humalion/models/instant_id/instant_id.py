import importlib
import os
import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import PIL
from huggingface_hub import hf_hub_download
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image

from tqdm.auto import tqdm

from src.humalion.engine.face_embedding_model import FaceEmbeddingModel
from src.humalion.models.instant_id.face_analysis import FixedFaceAnalysis
from src.humalion.utils.github import github_download_file
from src.humalion.utils.path import posix_to_pypath
from .pipeline_stable_diffusion_xl_instantid import draw_kps
from ...engine.face_swapper import FaceSwapperModel
from ...utils.mixins import SaveImageWithUniqueNameMixin


class InstantID(FaceSwapperModel, SaveImageWithUniqueNameMixin):
    BASE_RESOURCE_DIR = Path('humalion_resources')
    CONTROLNET_DIR = Path("ControlNetModel")
    CONTROLNET_CONFIG_FILENAME = Path("config.json")
    CONTROLNET_CONFIG_FILEPATH = CONTROLNET_DIR / CONTROLNET_CONFIG_FILENAME
    CONTROLNET_WEIGHTS_FILENAME = Path("diffusion_pytorch_model.safetensors")
    CONTROLNET_WEIGHTS_FILEPATH = CONTROLNET_DIR / CONTROLNET_WEIGHTS_FILENAME
    IP_ADAPTER_FILENAME = Path("ip-adapter.bin")
    FILEPATHS_FOR_DOWNLOADING = {
        CONTROLNET_CONFIG_FILEPATH,
        CONTROLNET_WEIGHTS_FILEPATH,
        IP_ADAPTER_FILENAME,
    }
    UTILS_DIR = Path('instant_utils/')
    INSTANTID_GITHUB_REPO = 'InstantID/InstantID'
    IP_ADAPTER_GITHUB_PATH = Path('ip_adapter')
    IP_ADAPTER_ATTENTION_PROCESSOR_FILENAME = Path('attention_processor.py')
    IP_ADAPTER_RESAMPLER_FILENAME = Path('resampler.py')
    IP_ADAPTER_UTILS_FILENAME = Path('utils.py')
    IP_ADAPTER_FILES_LIST = [
        IP_ADAPTER_ATTENTION_PROCESSOR_FILENAME,
        IP_ADAPTER_RESAMPLER_FILENAME,
        IP_ADAPTER_UTILS_FILENAME
    ]
    DEFAULT_OUTPUT_DIR = 'output/instant_id'
    PORTRAIT_SIZE = (1280, 1280)
    PARENT_PATH = Path(__file__).resolve().parent
    MEN_POSES_PATH = PARENT_PATH / 'poses' / 'men.npy'
    WOMEN_POSES_PATH = PARENT_PATH / 'poses' / 'women.npy'
    REALIVIZ_NAME = "SG161222/RealVisXL_V4.0_Lightning"  # https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning
    PIPELINE_NUM_STEPS = 8

    def __init__(
            self,
            models_weight_path: str = './checkpoints',
            output_dir: str = DEFAULT_OUTPUT_DIR,
            use_cuda: bool = False,
            negative_prompt: str = ''
    ):
        self.use_cuda = use_cuda
        self.models_weight_path = models_weight_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_weight_path, exist_ok=True)
        os.makedirs(self.BASE_RESOURCE_DIR, exist_ok=True)
        (self.BASE_RESOURCE_DIR / "__init__.py").touch(exist_ok=True)

        self.face_detector = None
        self.controlnet = None
        self._pipeline = None
        self._poses_men = None
        self._poses_women = None
        self.negative_prompt = negative_prompt
        self._pbar = tqdm(total=100)
        self.prepare_model()

    def _check_or_download_models_files(self, force_download: bool = False):
        bar_desc = "Downloading Models weight... It may take be a long process at first time, relax and wait..."
        for fp in tqdm(self.FILEPATHS_FOR_DOWNLOADING, desc=bar_desc, leave=False):
            if not fp.is_file() or force_download:
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename=fp.as_posix(),
                    local_dir=self.models_weight_path
                )

    @classmethod
    def _download_instantid_repo(cls, force_download: bool = False):
        utils_path = (cls.BASE_RESOURCE_DIR / cls.UTILS_DIR)
        os.makedirs(utils_path, exist_ok=True)

        sub_prbar = tqdm(total=2, leave=False)
        sub_prbar.set_description("Downloading InstantID utils files...")

        # download InstantID repo files
        ip_adapter_path = cls.BASE_RESOURCE_DIR / cls.UTILS_DIR / cls.IP_ADAPTER_GITHUB_PATH
        os.makedirs(ip_adapter_path, exist_ok=True)
        for ip_adapter_filename in cls.IP_ADAPTER_FILES_LIST:
            filepath = utils_path / cls.IP_ADAPTER_GITHUB_PATH / ip_adapter_filename
            github_filepath = cls.IP_ADAPTER_GITHUB_PATH / ip_adapter_filename
            if not filepath.is_file() or force_download:
                print(ip_adapter_path.as_posix(), cls.INSTANTID_GITHUB_REPO, cls.IP_ADAPTER_GITHUB_PATH.as_posix())
                github_download_file(
                    download_dir=(utils_path / cls.IP_ADAPTER_GITHUB_PATH).as_posix(),
                    repo=cls.INSTANTID_GITHUB_REPO,
                    path=github_filepath.as_posix()
                )
        sub_prbar.update(1)

        # create __init__.py files
        init_filename = "__init__.py"
        (utils_path / init_filename).touch(exist_ok=True)
        (utils_path / cls.IP_ADAPTER_GITHUB_PATH / init_filename).touch(exist_ok=True)

        sub_prbar.set_description("Importing utils files...")
        for ip_adapter_filename in cls.IP_ADAPTER_FILES_LIST:
            importlib.import_module(
                name=f'{posix_to_pypath(utils_path)}.{cls.IP_ADAPTER_GITHUB_PATH}.{ip_adapter_filename.stem}')
        sub_prbar.update(2)

    def _load_poses(self):
        self._poses_men = np.load(self.MEN_POSES_PATH)
        self._poses_women = np.load(self.WOMEN_POSES_PATH)

    def prepare_model(self, force_download: bool = False):
        self._pbar.set_description("Preparing InstantID...")
        self._check_or_download_models_files(force_download)
        self._pbar.update(10)

        self._download_instantid_repo()
        self._pbar.update(15)

        self._pbar.set_description("Downloading and preparing FaceAnalysis...")
        self.face_detector = FixedFaceAnalysis(
            name='antelopev2', root=self.BASE_RESOURCE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._pbar.update(30)
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self._pbar.set_description("Preparing ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(self.models_weight_path, self.CONTROLNET_DIR),
            torch_dtype=torch.float16
        )
        self._pbar.update(40)

        self._pbar.set_description("Downloading RealVisXL_V4.0_Lightning...")

        from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        self._pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
            pretrained_model_name_or_path=self.REALIVIZ_NAME,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )
        self._pbar.update(90)
        self._pbar.set_description("Loading models to memory...")
        if self.use_cuda and torch.cuda.is_available():
            self._pipeline.cuda()

        # load adapter
        self._pipeline.load_ip_adapter_instantid(os.path.join(self.models_weight_path, self.IP_ADAPTER_FILENAME))
        self._pbar.update(100)

    def swap_face_from_docs(self, source_image_path, face_emb: np.ndarray) -> str:
        from .pipeline_stable_diffusion_xl_instantid import draw_kps

        source_image = load_image(source_image_path)

        # prepare face emb
        face_info = self.face_detector.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(source_image, face_info['kps'])  # noqa: F821
        prompt = "highly detailed, sharp focus, ultra sharpness, cinematic"
        negative_prompt = ""

        image: PIL.Image = self._pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
        ).images[0]
        file_ext = '.jpg'
        filepath = Path(self.output_dir) / (uuid.uuid4().hex + file_ext)
        image.save(filepath)
        return filepath.as_posix()

    def _random_pose(self, gender: FaceSwapperModel.Gender) -> np.ndarray:
        if gender == FaceSwapperModel.Gender.FEMALE:
            idx = random.randint(0, self._poses_women.shape[0] - 1)
            kps = self._poses_women[idx]
        else:
            idx = random.randint(0, self._poses_men.shape[0] - 1)
            kps = self._poses_men[idx]

        return kps

    def swap_face(
            self,
            source_image_path,
            face_emb: np.ndarray,
            prompt: str,
            gender: FaceSwapperModel.Gender | str
    ) -> str:
        if isinstance(gender, str):
            if gender not in FaceSwapperModel.Gender:
                raise ValueError(f"Gender must be one of {[*FaceSwapperModel.Gender]}")
            else:
                gender = FaceSwapperModel.Gender(gender)

        empty_img = Image.new("RGB", self.PORTRAIT_SIZE, (0, 0, 0))
        pose = self._random_pose(gender=gender)
        face_kps = draw_kps(empty_img, pose)
        self._pipeline.set_ip_adapter_scale(0.8)
        pipeline_result = self._pipeline(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=self.PIPELINE_NUM_STEPS,
            guidance_scale=3.0,
        )

        if not pipeline_result:
            raise ValueError(f"Error occured while swapping face on source image {source_image_path}")

        image = pipeline_result[0]
        image_path = self._save_image(self.output_dir, image)
        return image_path