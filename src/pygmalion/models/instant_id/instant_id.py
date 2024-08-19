import os
from abc import ABC, abstractmethod
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

from pygmalion.engine.face_embedding_model import FaceEmbeddingModel


class InstantID(FaceEmbeddingModel):
    CONTROLNET_DIR = "ControlNetModel/"
    CONTROLNET_CONFIG_FILENAME = "config.json"
    CONTROLNET_CONFIG_FILEPATH = os.path.join(CONTROLNET_DIR, CONTROLNET_CONFIG_FILENAME)
    CONTROLNET_WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"
    CONTROLNET_WEIGHTS_FILEPATH = os.path.join(CONTROLNET_DIR, CONTROLNET_WEIGHTS_FILENAME)
    IP_ADAPTER_FILENAME = "ip-adapter.bin"
    FILEPATHS_FOR_DOWNLOADING = {
        CONTROLNET_CONFIG_FILEPATH,
        CONTROLNET_WEIGHTS_FILEPATH,
        IP_ADAPTER_FILENAME,
    }

    def __init__(self, models_weight_path: str = './checkpoints'):
        self.models_weight_path = models_weight_path
        self.face_detector = None
        self.controlnet = None
        self.pipeline = None
        self.prepare_model()

    def check_or_download_models_files(self, force_download: bool = False):
        for fp in self.FILEPATHS_FOR_DOWNLOADING:
            if not os.path.isfile(fp) or force_download:
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename=fp,
                    local_dir=self.models_weight_path
                )

    def prepare_model(self, force_download: bool = False):
        self.check_or_download_models_files(force_download)

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
