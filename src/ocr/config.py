"""Hazard-agnostic storage config. Fire still uses its own config! TODO: refactor onto this later"""

import re
from dataclasses import dataclass
from enum import StrEnum
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


class Hazard(StrEnum):
    FIRE = 'fire'
    FLOOD = 'flood'
    HURRICANE = 'hurricane'


@dataclass(frozen=True)
class OutputPaths:
    """Bucket-relative output keys for one hazard version. Resolve with StorageConfig.uri / icechunk_storage.

    v2+ layout only; fire v1.x stays at fire's legacy prefixes (output/fire-risk/...).

    >>> cfg, out = StorageConfig(), OutputPaths(Hazard.FLOOD, 'v2.0.0')
    >>> repo = icechunk.Repository.open(cfg.icechunk_storage(out.tensor))
    >>> write_geoparquet(tbl, cfg.uri(out.buildings))
    """

    hazard: Hazard
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'hazard', Hazard(self.hazard))
        if not re.fullmatch(r'v\d+\.\d+\.\d+', self.version):
            raise ValueError(f'version must look like v2.0.0, got {self.version!r}')

    @property
    def root(self) -> str:
        return f'output/{self.hazard}/{self.version}'

    @property
    def tensor(self) -> str:
        return f'{self.root}/tensor.icechunk'

    @property
    def buildings(self) -> str:
        return f'{self.root}/buildings.parquet'

    @property
    def regions(self) -> str:
        """Prefix for `ocr.regions.layout` (nation/states files, per-state counties/tracts/blocks)."""
        return f'{self.root}/regions'
