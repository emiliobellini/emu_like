"""Joined targets must preserve physical spectra without new CLASS runs."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
from emu_like.datasets import Dataset, SobolevDataset
from emu_like.y_models import ClassSpectra
from test_sampling_redshift_coverage import CoverageTests


def dataset(limit, scale, kind=Dataset, n=5, ratio=True):
    m = CoverageTests().model(configured=limit, growth=False)['pk_m']
    m.hiclassy = None
    m.cosmo = None
    # The wider grid is deliberately sparser: selection must use coverage.
    m.z_array = np.linspace(0, limit, 5 if limit == 2 else 11)
    m.spectra[0].ratio = ratio
    m.outputs['pk_m']['ratio'] = ratio
    z = m.z_array
    m.y_ref = [np.broadcast_to(scale*(2+z+z*z), (1, 5, len(z))).copy()]
    if not ratio:
        m.y_ref[0][:] = 1
    m.n_samples = n
    x = np.linspace(0, limit, n)[:, None]
    physical = np.broadcast_to(10+x, (n, 5)).copy()
    m.y = [physical / (scale*(2+x+x*x)) if ratio else physical]
    d = kind(name='pk_m', x=x, y=m.y[0], x_names=['z_pk'],
             y_names=m.y_names[0], y_model=m, path=f'{limit}.fits',
             y_header=m.y_headers[0], x_key='x_data')
    if kind is SobolevDataset:
        d.growth_name = 'fk_m'
        d.growth_y_names = ['f']*5
        d.y_growth = np.full((n, 5), .7)
        d.reference_pk = m.y_ref[0].copy()
        d.redshift_grid = m.z_array.copy()
        d.reference_growth = kind.growth_from_reference_pk(d.reference_pk, z)
    return d


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('kind', [Dataset, SobolevDataset])
def test_join_preserves_physical_spectra_and_input_arrays(reverse, kind):
    small = dataset(1., 1.2, kind, n=1030)
    large = dataset(2., 1., kind)
    datasets = [large, small] if reverse else [small, large]
    original = [copy.deepcopy(d) for d in datasets]
    with patch.object(ClassSpectra, 'evaluate',
                      side_effect=AssertionError('CLASS run')):
        joined = kind.join(datasets)
    np.testing.assert_array_equal(
        joined.y_model.z_array, large.y_model.z_array)
    np.testing.assert_array_equal(
        joined.y_model.spectra[0].z_array, large.y_model.z_array)
    x = joined.x
    np.testing.assert_allclose(
        joined.y*(2+x+x*x), np.broadcast_to(10+x, joined.y.shape), rtol=1e-13)
    np.testing.assert_array_equal(joined.y, joined.y_model.y[0])
    for d, old in zip(datasets, original):
        np.testing.assert_array_equal(d.y, old.y)
        np.testing.assert_array_equal(d.y_model.y_ref[0], old.y_model.y_ref[0])
    if kind is SobolevDataset:
        np.testing.assert_array_equal(
            joined.y_growth, np.vstack([d.y_growth for d in datasets]))
        np.testing.assert_array_equal(
            joined.reference_pk, joined.y_model.y_ref[0])
        np.testing.assert_array_equal(
            joined.redshift_grid, large.redshift_grid)
        z = joined.redshift_grid
        expected = -.5*(1+z)*(1+2*z)/(2+z+z*z)
        np.testing.assert_allclose(joined.reference_growth, np.broadcast_to(
            expected, joined.reference_growth.shape), rtol=1e-13)


def test_unnormalized_targets_unchanged():
    inputs = [dataset(1., 1., ratio=False), dataset(2., 1., ratio=False)]
    joined = Dataset.join(inputs)
    np.testing.assert_array_equal(joined.y, np.vstack([d.y for d in inputs]))


@pytest.mark.parametrize('bad', ['outside', 'zero', 'nan', 'parameters'])
def test_join_rejects_invalid_normalization_inputs(bad):
    a, b = dataset(1., 1.2), dataset(2., 1.)
    if bad == 'outside':
        a.x[0, 0] = 1.5
    if bad == 'zero':
        b.y_model.y_ref[0][:] = 0
    if bad == 'nan':
        a.y_model.y_ref[0][0, 0, 0] = np.nan
    if bad == 'parameters':
        a.y_model.ref_params['h'] += .1
    with pytest.raises(ValueError):
        Dataset.join([a, b])
