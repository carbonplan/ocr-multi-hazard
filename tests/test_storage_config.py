from pathlib import Path

import pytest

from ocr.config import OutputPaths, Risk, StorageConfig


def test_output_paths():
    out = OutputPaths('hurricane', 'v2.1.0')
    assert out.risk is Risk.HURRICANE
    assert out.tensor == 'output/hurricane/v2.1.0/tensor.icechunk'
    assert out.buildings == 'output/hurricane/v2.1.0/buildings.parquet'
    assert out.regions == 'output/hurricane/v2.1.0/regions'


def test_output_paths_resolve(tmp_path: Path):
    out = OutputPaths(Risk.FLOOD, 'v2.0.0')
    s3 = StorageConfig(bucket='other-bucket', local_root=None)
    assert s3.uri(out.buildings) == 's3://other-bucket/output/flood/v2.0.0/buildings.parquet'
    local = StorageConfig(local_root=tmp_path)
    assert local.uri(out.tensor) == str(tmp_path / 'output/flood/v2.0.0/tensor.icechunk')
    assert local.uri(out.regions) == str(tmp_path / 'output/flood/v2.0.0/regions')


@pytest.mark.parametrize('version', ['2.0.0', 'v2', 'v2.0', 'v2.0.0-dev'])
def test_bad_version(version: str):
    with pytest.raises(ValueError):
        OutputPaths(Risk.FIRE, version)


def test_bad_risk():
    with pytest.raises(ValueError):
        OutputPaths('tornado', 'v2.0.0')
