"""Numerical checks for the joint power/growth diagnostic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
# Load emulator helpers only after the optional TensorFlow dependency check.
scalers = importlib.import_module('emu_like.scalers')
StandardScaler = scalers.StandardScaler
LogStandardScaler = scalers.LogStandardScaler
PCA = importlib.import_module('emu_like.pca').PCA

# Import the command-line script without invoking its main pipeline.
spec = importlib.util.spec_from_file_location(
    'compare_pk_fk',
    Path(__file__).parents[1]/'scripts/check_train/compare_pk_fk.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


@pytest.fixture(autouse=True)
def double_precision_policy():
    # Keep finite-difference assertions sensitive to derivative mistakes,
    # rather than float32 rounding, and restore the caller's policy afterward.
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float64')
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def test_pca_scaling_derivative_matches_numpy_and_finite_difference():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 3))
    y = np.exp(rng.normal(size=(100, 4)))
    sx, sy = StandardScaler(), LogStandardScaler()
    sx.fit(x)
    sy.fit(y)
    px, py = PCA(3), PCA(4)
    px.fit(sx.transform(x))
    py.fit(sy.transform(y))
    model = tf.keras.Sequential([tf.keras.layers.Input((3,), dtype='float64'),
                                tf.keras.layers.Dense(4, dtype='float64')])
    model.set_weights([rng.normal(size=(3, 4))*.1, np.zeros(4)])
    emu = SimpleNamespace(
        model=model, x_scaler=sx, y_scaler=sy, x_pca=px, y_pca=py)
    xx = x[:6]
    value, dz = diag.value_and_dz(emu, xx, 1)
    expected = sy.inverse_transform(py.inverse_transform(
        model(px.transform(sx.transform(xx))).numpy()))
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    plus, minus = xx.copy(), xx.copy()
    plus[:, 1] += 1e-5
    minus[:, 1] -= 1e-5
    vp = diag.value_and_dz(emu, plus, 1)[0]
    vm = diag.value_and_dz(emu, minus, 1)[0]
    np.testing.assert_allclose(dz, (vp-vm)/2e-5, atol=1e-9)


def test_reference_derivative_and_boundary_stencils():
    model = tf.keras.Sequential([tf.keras.layers.Input((1,), dtype='float64'),
                                tf.keras.layers.Dense(2, dtype='float64')])
    model.set_weights([np.array([[.3, -.2]]), np.array([1., 2.])])
    emu = SimpleNamespace(
        model=model, x_scaler=None, y_scaler=None,
        x_pca=None, y_pca=None, y_names=['a', 'b'])
    grid = np.linspace(0, 2, 10)
    ref = diag.spline(grid, np.stack([2+grid**2, 3+grid**2]))
    x = np.array([[0.], [.7], [2.]])
    p, dp = diag.value_and_dz(emu, x, 0)
    # Linear network targets and a quadratic normalization give an analytic
    # growth value at both boundaries and at an interior redshift.
    expected = -.5*(1+x)*(
        np.array([.3, -.2])/p + 2*x/(np.array([2., 3.])+x**2))
    np.testing.assert_allclose(
        diag.growth_from_ratio(p, dp, x[:, 0], ref), expected, atol=1e-12)
    fd = diag.finite_difference_growth(emu, x, ref, 0, .0001, (0, 2))
    np.testing.assert_allclose(fd, expected, atol=1e-7)


def test_missing_relative_bins_and_empty_groups_are_reported():
    r = diag.summarize(np.array([[np.nan, 1.], [2., 3.]]))
    assert r['bins']['nonfinite'] == 1
    assert r['complete_spectra'] == 1
    assert r['spectrum_max_abs']['max_abs'] == 3
    assert diag.summarize(np.empty((0, 3)))['bins']['count'] == 0


def test_validation_intersection_uses_each_models_own_filter(tmp_path):
    from astropy.io import fits
    from sklearn.model_selection import train_test_split
    path = tmp_path/'sample.fits'
    x = np.arange(120, dtype=float).reshape(40, 3)
    p, f = np.ones((40, 2)), np.ones((40, 2))
    p[1] = np.nan
    f[7] = np.nan
    fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(x, name='X_DATA'),
        fits.ImageHDU(p, name='PK_M'), fits.ImageHDU(f, name='FK_M')
    ]).writeto(path)
    configs = [{'datasets': dict(
        paths=[str(path)], remove_non_finite=True,
        frac_train=.5, train_test_random_seed=15)} for _ in range(3)]
    _, sizes, ids = diag.common_rows(configs, 'pk_m', 'fk_m')
    expected = []
    # Different invalid target rows must be removed before each split.
    for bad in [[1], [7], [1, 7]]:
        valid = np.setdiff1d(np.arange(40), bad)
        expected.append(train_test_split(
            valid, train_size=.5, random_state=15)[1])
    np.testing.assert_array_equal(
        ids, np.intersect1d(np.intersect1d(*expected[:2]), expected[2]))
    assert sizes == [40]
