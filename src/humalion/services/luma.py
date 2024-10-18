import os
import uuid

from lumaai import Lumaai

from src.humalion.utils.mixins import CheckOrCreateDirMixin, DownloadFileMixin


class LumaService(CheckOrCreateDirMixin, DownloadFileMixin):
    output_dir = "output/luma"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._check_dir(self.output_dir)
        self.api_key = api_key
        self.client = Lumaai(auth_token=self.api_key)

    def _save_video(self, url: str) -> str:
        filename = f"{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        self._download_file(url=url, download_path=filepath)
        return filepath
