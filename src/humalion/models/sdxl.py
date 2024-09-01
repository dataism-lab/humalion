import base64
import os
import uuid

import requests
from requests import Response

from src.humalion.engine.generative_model import GenerativeModel

from src.humalion.services.stability_ai import StabilityAI


class SDXL(StabilityAI, GenerativeModel):
    BASE_PROMPT_PREFIX = StabilityAI.BASE_PROMPT_PREFIX
    BASE_PROMPT_SUFFIX = StabilityAI.BASE_PROMPT_SUFFIX
    ENGINE_ID = "stable-diffusion-v1-6"
    _text_to_img_url = f"{StabilityAI.API_HOST}/v1/generation/{ENGINE_ID}/text-to-image"
    AVAILABLE_RESOLUTIONS = {
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536)
    }

    def __init__(
            self,
            api_key: str,
            base_prompt_prefix: str | None = None,
            base_prompt_suffix: str | None = None,
            output_resolution: list[int, int] | tuple[int, int] = (1024, 1024),
            model_scale: int = 25,
            output_dir: str = "output/sdxl/",
            **kwargs,
    ):
        if output_resolution not in self.AVAILABLE_RESOLUTIONS:
            raise ValueError(f"output_resolution must be one of the following: {self.AVAILABLE_RESOLUTIONS}")

        self.model_scale = model_scale
        self.output_resolution = output_resolution
        self.additional_model_params = kwargs
        super().__init__(
            api_key=api_key,
            base_prompt_prefix=base_prompt_prefix,
            base_prompt_suffix=base_prompt_suffix,
            output_dir=output_dir
        )

    def _generate_response(self, data: dict):
        return requests.post(url=self._text_to_img_url, headers=self.headers, json=data)

    def _save_img(self, response: Response) -> str:
        """Save an image from the given response.

        Args:
            response (Response): The response object containing the image data.

        Returns:
            str: The filepath of the saved image.

        """
        resp_data = response.json()
        filename = f'{uuid.uuid4().hex}.png'
        filepath = os.path.join(self.output_dir, filename)
        image = resp_data["artifacts"][0]
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image["base64"]))

        return filepath


    def _request_data(self, prompt: str) -> dict:
        request_data = {
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": self.model_scale,
            "height": self.output_resolution[1],
            "width": self.output_resolution[0],
            "samples": 1,
            "steps": 30,
            "style_preset": "photographic",
            **self.additional_model_params
        }
        return request_data
