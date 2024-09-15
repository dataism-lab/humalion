import os

from requests import Response

from src.humalion.engine.generative_model import GenerativeModel
from abc import ABC, abstractmethod, ABCMeta
import requests

from src.humalion.services.mixins import CheckOrCreateDirMixin


class StabilityAI(GenerativeModel, CheckOrCreateDirMixin, ABC, metaclass=ABCMeta):
    BASE_PROMPT_PREFIX = "High-definition, full-body portrait photograph of a "
    BASE_PROMPT_SUFFIX = """ suitable for a popular Instagram post. 
        Cinematic composition, professional color grading, film grain, atmospheric."""
    API_HOST = 'https://api.stability.ai'

    def __init__(
            self,
            api_key: str,
            base_prompt_prefix: str | None = None,
            base_prompt_suffix: str | None = None,
            output_dir: str = "output/stability/"
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
        self.output_dir = output_dir
        self._check_dir(self.output_dir)

    def _cover_prompt(self, prompt: str) -> str:
        return f'{self.base_prompt_prefix} {prompt} {self.base_prompt_suffix}'

    @abstractmethod
    def _request_data(self, prompt: str) -> dict:
        pass

    @abstractmethod
    def _generate_response(self, data: dict, *args, **kwargs):
        pass

    @abstractmethod
    def _save_img(self, response: Response):
        pass

    def _stability_generate_photo(self, prompt: str) -> str:
        data = self._request_data(prompt=prompt)
        response = self._generate_response(data)

        if response.status_code != 200:
            raise requests.exceptions.RequestException(f"Stability API request failed: {str(response.text)}")

        filepath = self._save_img(response)

        return filepath

    def generate_photo(self, prompt: str) -> str:
        prompt = self._cover_prompt(prompt)
        photo_path = self._stability_generate_photo(prompt)
        return photo_path

    def prepare_model(self):
        pass