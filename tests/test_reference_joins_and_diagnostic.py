"""Join equivalent reference splines and compare multi-z CLASS accessors."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from emu_like.y_models import ClassSpectra
from emu_like.spectra import Spectra


def model(z):
    spectra = Spectra({'pk_m': dict(k_min=.01, k_max=1., k_num=4,
                                    k_space='log', ratio=True)})
    return SimpleNamespace(
        spectra=spectra, z_array=z,
        y_ref=[np.broadcast_to(
            (2+z+z**2)[None, None, :], (1, 4, len(z))).copy()],
        ref_params={'z_max_pk': z[-1]+.001}, _get_z_max=lambda: z[-1])


def test_join_different_nodes_and_explicit_endpoints():
    small = model(np.array([0., .2, .5, .8, 1.]))
    large = model(np.array([0., .3, .6, 1.2, 1.6, 2.]))
    for models in ([small, large], [large, small]):
        z, ref = ClassSpectra._join_references(models, large.ref_params)
        np.testing.assert_array_equal(z, large.z_array)
        np.testing.assert_array_equal(ref[0], large.y_ref[0])
        assert not np.shares_memory(ref[0], large.y_ref[0])


def test_join_does_not_silently_change_normalization():
    small = model(np.linspace(0., 1., 5))
    large = model(np.linspace(0., 2., 7))
    small.y_ref[0] *= 1.01
    with pytest.raises(ValueError, match='normalizations differ'):
        ClassSpectra._join_references([small, large], large.ref_params)


def test_join_compares_between_matching_nodes():
    small = model(np.array([0., .25, .5, .75, 1.]))
    large = model(np.array([0., .25, .5, .75, 1., 1.5, 2.]))
    large.y_ref[0][..., 5] += .5
    with pytest.raises(ValueError, match='normalizations differ'):
        ClassSpectra._join_references([small, large], large.ref_params)


def test_full_join_with_different_reference_limits():
    import test_sampling_redshift_coverage as coverage
    small = coverage.CoverageTests().model(configured=1.)
    large = coverage.CoverageTests().model(configured=2.)
    large.classy = small.classy
    joined = ClassSpectra.join([small, large])
    assert joined.n_samples == 4
    assert joined.z_array[-1] == 2.
    assert joined.ref_params['z_max_pk'] == 2.001
    for values in joined.y:
        assert values.shape == (4, 5)


@pytest.mark.parametrize('grid', [np.array([0., .5, .3, 1.]),
                                  np.array([0., .5, np.nan, 1.])])
def test_join_rejects_invalid_grid(grid):
    m = model(grid)
    with pytest.raises(ValueError, match='finite, increasing'):
        ClassSpectra._join_references([m], m.ref_params)


def test_diagnostic_array_evaluator_fills_every_redshift():
    spec = importlib.util.spec_from_file_location(
        'check_class',
        Path(__file__).parents[1]/'scripts/check_data/check_class.py')
    diagnostic = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'classy': SimpleNamespace()}):
        spec.loader.exec_module(diagnostic)

    class FakeClass:
        def h(self):
            return .7

        def pk_lin(self, k, z):
            return k*(1+z)**2

        def get_pk_lin(self, k, z, n_k, n_z, n_mu):
            assert np.all(k > 0)
            return k*(1+z[None, :, None])**2

    c = FakeClass()
    k, z = np.array([.01, .1, 1.]), np.array([1., 0., .5, 1.])
    np.testing.assert_allclose(diagnostic.get_pk_3(c, k, z),
                               diagnostic.get_pk_2(c, k, z))
