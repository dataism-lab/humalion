import os
import uuid
from typing import Any

import replicate
from PIL import Image

from ..utils.mixins import DownloadTempFileMixin


class ReplicaService(DownloadTempFileMixin):
    output_dir = "output/replica"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._check_dir(self.output_dir)
        self.api_key = api_key
        self._export_api_key()

    def _export_api_key(self):
        os.environ["REPLICATE_API_TOKEN"] = self.api_key

    def _save_img(self, url: str) -> str:
        """Save an image from the given response.

        Args:
            url: url of .webp image

        Returns:
            str: The filepath of the saved .jpeg image

        """
        filename = f"{uuid.uuid4().hex}.jpeg"
        filepath = os.path.join(self.output_dir, filename)
        tmp_webp_file_path = self._download_tmp_file(url=url)
        img = Image.open(tmp_webp_file_path).convert("RGB")
        img.save(filepath, format="jpeg")
        os.remove(tmp_webp_file_path)

        return filepath

    def run_model(self, model: str, input_data: dict) -> Any:
        output = replicate.run(model, input=input_data)
        if not output:
            raise Exception("Failed to run replica API model")

        filepath = self._save_img(output[0])
        return filepath
