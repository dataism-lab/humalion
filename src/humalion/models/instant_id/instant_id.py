import os
from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import hf_hub_download
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

from humalion.engine.face_embedding_model import FaceEmbeddingModel
from humalion.utils.github import github_download_file


class InstantID(FaceEmbeddingModel):
    BASE_RESOURCE_DIR = Path('.humalion_resourses')
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
    UTILS_DIR = Path('instant-utils/')
    INSTANTID_GITHUB_REPO = 'InstantID/InstantID'
    INSTANTID_PIPLINE_FILENAME = Path('pipeline_stable_diffusion_xl_instantid.py')

    def __init__(self, models_weight_path: str = './checkpoints'):
        self.models_weight_path = models_weight_path
        self.face_detector = None
        self.controlnet = None
        self.pipeline = None
        self.prepare_model()

    def check_or_download_models_files(self, force_download: bool = False):
        for fp in self.FILEPATHS_FOR_DOWNLOADING:
            if not fp.is_file() or force_download:
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename=fp.as_posix(),
                    local_dir=self.models_weight_path
                )

    @classmethod
    def download_instant_utils(cls):
        github_download_file(
            download_dir=(cls.BASE_RESOURCE_DIR / cls.UTILS_DIR).as_posix(),
            repo=cls.INSTANTID_GITHUB_REPO,
            path=cls.INSTANTID_PIPLINE_FILENAME.as_posix()
        )

    def prepare_model(self, force_download: bool = False):
        self.check_or_download_models_files(force_download)
        self.download_instant_utils()

        self.face_detector = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(self.models_weight_path, self.CONTROLNET_DIR),
            torch_dtype=torch.float16
        )

        base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
        self.pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )
        self.pipeline.cuda()

        # load adapter
        self.pipeline.load_ip_adapter_instantid(os.path.join(self.models_weight_path, self.IP_ADAPTER_FILENAME))

    def generate_embeddings(self, source_image_path):
        source_image = load_image(source_image_path)

        # prepare face emb
        face_info = self.face_detector.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(source_image, face_info['kps'])

        image = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
        ).images[0]
