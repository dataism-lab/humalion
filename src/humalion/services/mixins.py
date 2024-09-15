import os
import uuid
from pathlib import Path

import requests


class CheckOrCreateDirMixin:
    @staticmethod
    def _check_dir(target_dir: str | os.PathLike):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)


class DownloadFileMixin:
    @staticmethod
    def _download_file(url: str, download_path: str):
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            response.raise_for_status()

        with open(download_path, 'wb') as file:
            file.write(response.content)


class DownloadTempFileMixin(DownloadFileMixin, CheckOrCreateDirMixin):
    DOWNLOAD_DIR = Path(".tmp_download/")

    def __init__(self):
        self._check_dir(self.DOWNLOAD_DIR)

    def _download_tmp_file(self, url: str) -> str:
        file_name = f"{uuid.uuid4()}{Path(url).suffix}"
        file_path = self.DOWNLOAD_DIR / file_name
        self._download_file(url=url, download_path=file_path.as_posix())
        return file_path.as_posix()
