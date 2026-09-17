"""Regression coverage for the standalone dataset checker's reference grid."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.mark.parametrize('offset', [0., 0.1])
def test_reference_check_preserves_grid_and_derivative_coverage(offset):
    spec = importlib.util.spec_from_file_location(
        'sampler_check', Path(__file__).with_name('sampler.py'))
    checker = importlib.util.module_from_spec(spec)
    cosmo = Mock()
    with patch.dict('sys.modules', {
            'hiclassy': SimpleNamespace(HiClass=lambda: cosmo)}):
        spec.loader.exec_module(checker)

    grid = np.array([0., .3, .7, 1.])

    class Power:
        name = 'pk_m'
        is_pk = True
        ratio = True
        z_array = grid.copy()

        def get(self, cosmo, z=None):
            if z is None:
                # Native output includes extra redshifts and mutates metadata.
                self.z_array = np.linspace(0., 1.001, 8)
                return np.ones((2, 8))
            return np.array([1. + z, 2. + z])

    power = Power()
    unnormalized = SimpleNamespace(
        name='fk_m', is_pk=True, ratio=False,
        get=Mock(side_effect=AssertionError('No reference needed')))
    angular = SimpleNamespace(
        name='cl_tt', is_pk=False, ratio=True,
        get=Mock(return_value=np.array([2., 3.])))
    references = [np.array([[1. + grid, 2. + grid]]) + offset,
                  np.ones((1, 2, len(grid))), np.array([[2., 3.]])]
    params = {'z_max_pk': 1.001, 'output': 'mPk, wPk'}
    model = SimpleNamespace(
        spectra=[power, unnormalized, angular], y_ref=references,
        z_array=grid.copy(), ref_params=params.copy(), _get_z_max=lambda: 1.)
    originals = [a.copy() for a in references]
    with patch.object(checker.io, 'warning') as warning:
        checker._compare_reference_spectra(model, 1e-12)
    assert warning.call_count == int(offset != 0)
    cosmo.set.assert_called_once_with(params)
    cosmo.compute.assert_called_once_with()
    np.testing.assert_array_equal(power.z_array, grid)
    np.testing.assert_array_equal(model.z_array, grid)
    assert model.ref_params == params
    for actual, original in zip(references, originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize('configured', [None, 3.])
@pytest.mark.parametrize('growth', [False, True])
def test_sample_checker_matches_production_coverage_and_values(configured, growth):
    from test_sampling_redshift_coverage import CoverageTests, CoverageClass

    model = CoverageTests().model(configured=configured, growth=growth)
    spec = importlib.util.spec_from_file_location(
        'sampler_check', Path(__file__).with_name('sampler.py'))
    checker = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'hiclassy': model.hiclassy}):
        spec.loader.exec_module(checker)
    cosmo = CoverageClass()
    params = dict(model.class_params)
    # Exercise both normalization paths with nonconstant reference tables.
    for i, sp in enumerate(model.spectra):
        if i % 2 == 0:
            sp.ratio = True
            model.y_ref[i] *= 2. + model.z_array[None, None, :]
    # A decreasing sequence catches inherited coverage from earlier rows;
    # z=0 also exercises the forward derivative stencil.
    for z in [1., .2, 0.]:
        actual = checker._evaluate_with_class(model, [z], cosmo, params)
        expected = model.evaluate([z], 0)
        assert cosmo.computations[-1] == model.cosmo.computations[-1]
        for i, sp in enumerate(model.spectra):
            np.testing.assert_array_equal(actual[sp.name], expected[i][0])
