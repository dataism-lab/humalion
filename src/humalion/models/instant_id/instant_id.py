import importlib
import os
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
from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from ...engine.face_swapper import FaceSwapperModel


class InstantID(FaceSwapperModel):
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

    def __init__(
            self,
            models_weight_path: str = './checkpoints',
            output_dir: str = DEFAULT_OUTPUT_DIR,
            use_cuda: bool = False,
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
        self.pipeline = None
        self.pbar = tqdm(total=100)
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

    def prepare_model(self, force_download: bool = False):
        self.pbar.set_description("Preparing InstantID...")
        self._check_or_download_models_files(force_download)
        self.pbar.update(10)

        self._download_instantid_repo()
        self.pbar.update(15)

        self.pbar.set_description("Downloading and preparing FaceAnalysis...")
        self.face_detector = FixedFaceAnalysis(name='antelopev2', root=self.BASE_RESOURCE_DIR,
                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.pbar.update(30)
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self.pbar.set_description("Preparing ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(self.models_weight_path, self.CONTROLNET_DIR),
            torch_dtype=torch.float16
        )
        self.pbar.update(40)

        self.pbar.set_description("Downloading SDXL Unstable Diffusers by YamerMIX_v8'...")
        base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
        self.pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )
        self.pbar.update(90)
        self.pbar.set_description("Loading models to memory...")
        if self.use_cuda and torch.cuda.is_available():
            self.pipeline.cuda()

        # load adapter
        self.pipeline.load_ip_adapter_instantid(os.path.join(self.models_weight_path, self.IP_ADAPTER_FILENAME))
        self.pbar.update(100)

    def swap_face(self, source_image_path):
        source_image = load_image(source_image_path)

        # prepare face emb
        face_info = self.face_detector.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(source_image, face_info['kps'])  # noqa: F821
        prompt = "highly detailed, sharp focus, ultra sharpness, cinematic"
        negative_prompt = ""

        image: PIL.Image = self.pipeline(
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
        return filepath
