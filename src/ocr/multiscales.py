from collections.abc import Callable, Mapping
from typing import Any

import icechunk
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
import zarr
from topozarr import create_pyramid


def write_multiscales(
    repo: icechunk.Repository,
    crs: str,
    *,
    branch: str = 'main',
    variables: list[str] | None = None,
    message: str = 'write multiscales',
    overwrite: bool = False,
    derived: Mapping[str, Callable[[zarr.Group], None]] | None = None,
    **pyramid_kwargs: Any,
) -> str:
    """Write multiscale levels 1..N over the base raster in group "0" and commit.

    `overwrite` first drops the existing levels (numbered groups but "0"), so settings can
    change; other groups are kept.
    `derived` maps variables computed from a level's others to the function that writes one
    into a level group; those in group "0" are not coarsened, but written on each new level from
    its coarsened variables. Extra kwargs go to `topozarr.create_pyramid`. Returns the snapshot id.
    """
    session = repo.writable_session(branch)
    if overwrite:
        root = zarr.open_group(session.store, mode='r+')
        for name in [n for n in root.group_keys() if n.isdigit() and n != '0']:
            del root[name]
        root.attrs.pop('multiscales', None)
    ds = xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)
    ds = ds.proj.assign_crs(spatial_ref=crs)
    derived = {name: write for name, write in (derived or {}).items() if name in ds}
    # One variable per write: Pyramid.write looks up each variable's array while earlier
    # variables' writes are in flight, and on an S3 icechunk store that lookup blocks zarr's
    # event loop (IcechunkStore.__str__) that those writes need, so 2+ variables deadlock.
    levels = []
    for name in variables if variables is not None else list(ds.data_vars):
        if name in derived:
            continue
        pyramid = create_pyramid(ds[[name]], **pyramid_kwargs)
        levels = list(range(1, pyramid.levels))
        pyramid.write(session.store, mode='a', levels=levels)
    for level in levels:
        group = zarr.open_group(session.store, path=str(level), mode='r+')
        for write in derived.values():
            write(group)
    return session.commit(message)
