import os
import uuid

import requests
from requests import Response

from src.humalion.engine.generative_model import GenerativeModel

from src.humalion.services.stability_ai import StabilityAI


class SDCore(StabilityAI, GenerativeModel):
    BASE_PROMPT_PREFIX = StabilityAI.BASE_PROMPT_PREFIX
    BASE_PROMPT_SUFFIX = StabilityAI.BASE_PROMPT_SUFFIX
    _text_to_img_url = f"{StabilityAI.API_HOST}/v2beta/stable-image/generate/core"
    AVAILABLE_RATIO = {
        "16:9",
        "1:1",
        "21:9",
        "2:3",
        "3:2",
        "4:5",
        "5:4",
        "9:16",
        "9:21",
    }

    def __init__(
            self,
            api_key: str,
            output_ratio: str = "2:3",
            base_prompt_prefix: str | None = None,
            base_prompt_suffix: str | None = None,
            negative_prompt: str | None = None,
            output_dir: str = "output/sdcore/",
    ):
        if output_ratio not in self.AVAILABLE_RATIO:
            raise ValueError(f"output_ratio must be one of the following: {self.AVAILABLE_RATIO}")

        self.output_ratio = output_ratio
        self.negative_prompt = negative_prompt
        super().__init__(
            api_key=api_key,
            base_prompt_prefix=base_prompt_prefix,
            base_prompt_suffix=base_prompt_suffix,
            output_dir=output_dir
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*",
        }

    def _generate_response(self, data: dict):
        response = requests.post(url=self._text_to_img_url, headers=self.headers, files={"none": ''}, data=data)
        return response

    def _save_img(self, response: Response) -> str:
        """Save an image from the given response.

        Args:
            response (Response): The response object containing the image data.

        Returns:
            str: The filepath of the saved image.

        """
        filename = f'{uuid.uuid4().hex}.png'
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as file:
            file.write(response.content)

        return filepath

    def _request_data(self, prompt: str) -> dict:
        request_data = {
            "prompt": prompt,
            "aspect_ratio": self.output_ratio,
            "negative_prompt": self.negative_prompt,
            "output_format": "jpeg",
            "style_preset": "photographic"
        }
        return request_data
