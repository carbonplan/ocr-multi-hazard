"""Hazard-agnostic storage config. Fire still uses its own config! TODO: refactor onto this later"""

from pathlib import Path

import icechunk
import pydantic_settings
from obstore.store import LocalStore, S3Store


class StorageConfig(pydantic_settings.BaseSettings):
    bucket: str = 'carbonplan-ocr'
    region: str = 'us-west-2'
    local_root: Path | None = None

    model_config = {'env_prefix': 'ocr_storage_', 'case_sensitive': False}

    def object_store(self) -> LocalStore | S3Store:
        if self.local_root:
            return LocalStore(self.local_root.expanduser(), mkdir=True)
        return S3Store(self.bucket, region=self.region)

    def uri(self, key: str) -> str:
        if self.local_root:
            return str(self.local_root.expanduser() / key)
        return f's3://{self.bucket}/{key}'

    def icechunk_storage(self, prefix: str) -> icechunk.Storage:
        if self.local_root:
            return icechunk.local_filesystem_storage(str(self.local_root.expanduser() / prefix))
        return icechunk.s3_storage(bucket=self.bucket, prefix=prefix, region=self.region)
