import shutil
import uuid
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import aioboto3
import botocore
from loguru import logger


class AsyncS3MediaStorage:
    def __init__(self, connection_params: dict, bucket: str, temp_dir: str | PathLike):
        self.temp_dir = Path(temp_dir)
        self.session = aioboto3.Session()
        self.params = connection_params
        self.bucket = bucket

        self.check_and_clear_temp_dir()

    def check_and_clear_temp_dir(self):
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def upload_file(self, file_id: str, content: BinaryIO):
        async with self.session.client(**self.params) as s3:
            await s3.upload_fileobj(content, Bucket=self.bucket, Key=file_id)

    async def check_file_exist(self, file_id: str) -> bool:
        try:
            async with self.session.client(**self.params) as s3:
                await s3.head_object(Bucket=self.bucket, Key=file_id)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.exception(e)
                return False
            else:
                logger.exception(e)
                raise botocore.exceptions.DataNotFoundError(str(e)) from e
        else:
            return True

    async def download_file(self, file_id: str, file_path: PathLike, bucket: str | None = None):
        bucket = bucket if bucket is not None else self.bucket
        async with self.session.client(**self.params) as s3:
            await s3.download_file(Bucket=bucket, Key=file_id, Filename=str(file_path))

    async def download_temp_file(self, file_id: str, bucket: str | None = None) -> PathLike:
        """
        async download file to self.temp directory
        @param bucket: where to download the bucket from
        @param file_id: id of file in s3
        @return: path of temporary file
        """
        bucket = bucket if bucket is not None else self.bucket
        file_name = f"{uuid.uuid4()}{Path(file_id).suffix}"
        file_path = self.temp_dir / file_name
        await self.download_file(file_id=file_id, file_path=file_path, bucket=bucket)
        return file_path
