import base64
import uuid
import os

from humalion.engine.generative_model import GenerativeModel
import requests


class SDXL(GenerativeModel):
    BASE_PROMPT_PREFIX = "High-definition, full-body portrait photograph of a "
    BASE_PROMPT_SUFFIX = """ suitable for a popular Instagram post. 
    Cinematic composition, professional color grading, film grain, atmospheric."""
    ENGINE_ID = "stable-diffusion-v1-6"
    API_HOST = 'https://api.stability.ai'
    text_to_img_url = f"{API_HOST}/v1/generation/{ENGINE_ID}/text-to-image"
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
        super().__init__()
        self.api_key = api_key
        self.base_prompt_prefix = base_prompt_prefix if base_prompt_prefix is not None else self.BASE_PROMPT_PREFIX
        self.base_prompt_suffix = base_prompt_suffix if base_prompt_suffix is not None else self.BASE_PROMPT_SUFFIX
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        if output_resolution not in self.AVAILABLE_RESOLUTIONS:
            raise ValueError(f"output_resolution must be one of the following: {self.AVAILABLE_RESOLUTIONS}")
        self.output_resolution = output_resolution
        self.model_scale = model_scale
        self.additional_model_params = kwargs
        self.output_dir = output_dir
        self._check_output_dir()

    def _check_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _cover_prompt(self, prompt: str) -> str:
        return f'{self.base_prompt_prefix} {prompt} {self.base_prompt_suffix}'

    def _generate_photo_with_sdxl_api(self, prompt: str) -> str:
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
        print(request_data)
        response = requests.post(url=self.text_to_img_url, headers=self.headers, json=request_data)

        if response.status_code != 200:
            raise requests.exceptions.RequestException(f"Stability API request failed: {str(response.text)}")

        resp_data = response.json()
        filename = f'{uuid.uuid4().hex}.png'
        filepath = os.path.join(self.output_dir, filename)
        for i, image in enumerate(resp_data["artifacts"]):
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(image["base64"]))

        return filepath

    def generate_photo(self, prompt: str) -> str:
        prompt = self._cover_prompt(prompt)
        photo_path = self._generate_photo_with_sdxl_api(prompt)
        return photo_path


    def prepare_model(self):
        pass