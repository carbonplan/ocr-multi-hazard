import numpy as np
import pytest

from ocr.risks.hurricane import chaz, refit, store


def tutorial_cached() -> bool:
    return (store.CACHE / store.TUTORIAL / chaz.gv.obpath).exists()


def test_refit_on_the_tutorial_environment_recovers_the_shipped_coefficients(monkeypatch):
    monkeypatch.setattr(store, 'OFFLINE', True)
    if not tutorial_cached():
        pytest.skip('needs the tutorial inputs in the local cache')
    got, ref = refit.fit(refit.tutorial_env()), refit.shipped()
    for tag in 'wl':
        np.testing.assert_allclose(got[tag]['params'], ref[tag]['params'], atol=0.06)
    errors = got['errors']
    assert np.isfinite(errors).sum() == got['w']['n'] + got['l']['n']
    assert abs(np.nanmean(errors)) < 0.1


def test_refit_intensity_needs_fine_land(monkeypatch):
    monkeypatch.setattr(chaz.gv, 'land', 'shared')
    with pytest.raises(ValueError):
        chaz.apply_options({'intensity': 'refit'})
    chaz.apply_options({'intensity': 'refit', 'land': 'fine'})
    assert chaz.gv.intensity == 'refit'
    chaz.apply_options({})
    assert chaz.gv.intensity == 'tutorial'


def test_refit_normalization_is_only_for_era5():
    assert chaz.meanstd_for(chaz.ERA5, 'model', 'refit') == refit.key(chaz.ERA5, refit.FILES[2])
    with pytest.raises(ValueError):
        chaz.meanstd_for(chaz.ERA5, 'env', 'refit')
    with pytest.raises(ValueError):
        chaz.meanstd_for('cmip6/UKESM1-0-LL/r1i1p1f2/historical', 'model', 'refit')
