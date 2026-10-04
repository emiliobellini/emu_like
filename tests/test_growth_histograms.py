"""Per-run growth diagnostics restore references and preserve sample axes."""
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from scripts.check_train import plot_histograms as hist


@pytest.fixture(autouse=True)
def double_precision():
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float64')
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize('sobolev', [False, True])
def test_physical_growth_accuracy_and_consistency(tmp_path, sobolev, monkeypatch):
    # P(z) = (1 + .2 z) (2 + z)^2, while the independent FITS f
    # normalization is 2. This catches omitted reference derivatives and
    # comparisons against normalized, rather than physical, growth targets.
    z = np.array([0., .5, 1.])
    x = z[:, None]
    expected = -.5 * (1 + x) * (.2 / (1 + .2*x) + 2 / (2 + x))
    expected = np.repeat(expected, 2, axis=1)
    truth = expected + .1
    truth[1, 0] = 0.  # Absolute errors remain valid at zero growth.
    model = tf.keras.Sequential([
        tf.keras.layers.Input((1,), dtype='float64'),
        tf.keras.layers.Dense(2, dtype='float64')])
    model.set_weights([np.full((1, 2), .2), np.ones(2)])
    k = np.array([.1, 1.])
    emu = SimpleNamespace(
        model=model, x_names=['z_pk'], x_scaler=None, y_scaler=None,
        x_pca=None, y_pca=None,
        y_model=SimpleNamespace(k_ranges=[k], z_array=z,
                                y_ref=[np.repeat((2+z[None, :])**2, 2, axis=0)]))
    emu.eval_fk = lambda xx: np.repeat(
        -.5*(1+xx)*(.2/(1+.2*xx)+2/(2+xx)) + .05, 2, axis=1)
    data = {
        'x_data': np.concatenate([x, [[np.nan]]]),
        'fk_m': np.concatenate([truth/2, [[1., 1.]]]),
        'K_RANGE_fk_m': k, 'Z_ARRAY': z, 'REF_fk_m': np.full((2, 3), 2.),
    }
    dataset = SimpleNamespace(get_data=data.__getitem__, path='synthetic.fits')
    emudata = SimpleNamespace(emu=emu, is_sobolev=sobolev)
    groups, counts = hist.growth_residuals(
        emudata, dataset, 'fk_m', batch_size=1)
    assert counts == dict(input_rows=4, excluded_rows=1, evaluated_rows=3)
    np.testing.assert_allclose(
        groups['accuracy_absolute']['fk_from_pk'], expected-truth, atol=1e-12)
    rel = groups['accuracy_relative_percent']['fk_from_pk']
    assert np.isnan(rel[1, 0])
    assert hist.growth_diagnostics.summarize(rel)['complete_spectra'] == 2
    if sobolev:
        np.testing.assert_allclose(
            groups['consistency_absolute']['eval_fk - fk_from_pk'], .05,
            atol=1e-12)
    else:
        assert 'consistency_absolute' not in groups
    hist.show_growth_histograms(
        tmp_path, emudata, dataset, 'fk_m', 'test', batch_size=1)
    assert len(list(tmp_path.glob('*.png'))) == (4 if sobolev else 2)
    assert not list(tmp_path.glob('*.json'))
    assert not list(tmp_path.glob('*.npz'))
    monkeypatch.setattr(hist.io, 'FitsFile', lambda path: dataset)
    validation = hist.ValidationFits('synthetic.fits', np.array([2]), {'fk_m'})
    selected, counts = hist.growth_residuals(
        emudata, validation, 'fk_m', batch_size=1)
    assert counts == dict(input_rows=1, excluded_rows=0, evaluated_rows=1)
    for kind, residuals in selected.items():
        for name, values in residuals.items():
            np.testing.assert_allclose(values, groups[kind][name][[2]])


def test_empty_and_exact_zero_histograms(tmp_path):
    # No positive values can be put on a logarithmic axis; these cases
    # should still produce a figure and retain zeros in the statistics.
    path = tmp_path/'zeros.png'
    hist.growth_diagnostics.plot_histograms(
        {'zero': np.zeros((1, 2)), 'empty': np.empty((0, 2))}, path, 'test')
    assert path.is_file()
    assert hist.growth_diagnostics.summarize(
        np.zeros((1, 2)))['spectrum_rms']['rms'] == 0.


@pytest.mark.parametrize('sobolev', [False, True])
@pytest.mark.parametrize('filter_invalid', [False, True])
def test_validation_rows_match_joined_training_split(
        monkeypatch, sobolev, filter_invalid):
    from emu_like.datasets import Dataset, SobolevDataset

    inputs = {}
    for path, offset in [('thin', 0), ('ext', 20)]:
        x = np.arange(offset, offset + 20, dtype=float)[:, None]
        y = np.ones((20, 2))
        growth = np.ones((20, 2))
        y[2] = np.nan
        growth[5] = np.inf
        x[9] = np.nan
        inputs[path] = (x, y, growth)

    def load(self, path, **kwargs):
        self.x, self.y, self.y_growth = [a.copy() for a in inputs[path]]
        return self

    monkeypatch.setattr(Dataset, 'load', load)
    monkeypatch.setattr(SobolevDataset, 'load', load)
    config = dict(paths=['thin', 'ext'], name='pk_m',
                  remove_non_finite=filter_invalid, frac_train=.7,
                  train_test_random_seed=1543)
    params = dict(datasets=config, emulator=dict(
        name='sobolev_ffnn_emu' if sobolev else 'ffnn_emu'))
    selected = hist.validation_rows(params)

    # Run the training classes' filtering and splitting with original row
    # IDs as an extra finite input column, so membership can be compared.
    parts = []
    cls = SobolevDataset if sobolev else Dataset
    for i, arrays in enumerate(inputs.values()):
        x, y, growth = arrays
        data = cls(x=np.column_stack([x, np.arange(20) + i*20]), y=y.copy())
        if sobolev:
            data.y_growth = growth.copy()
        if filter_invalid:
            data.remove_non_finite()
        parts.append(data)
    joined = cls(x=np.concatenate([d.x for d in parts]),
                 y=np.concatenate([d.y for d in parts]))
    if sobolev:
        joined.y_growth = np.concatenate([d.y_growth for d in parts])
    joined.train_test_split(.7, 1543)
    actual = np.concatenate([rows + i*20 for i, rows in enumerate(selected)])
    np.testing.assert_array_equal(actual, np.sort(joined.x_test[:, -1]))
    assert not np.intersect1d(actual, joined.x_train[:, -1]).size


def test_unseeded_validation_is_rejected():
    with pytest.raises(ValueError, match='unseeded'):
        hist.validation_rows({'datasets': {'train_test_random_seed': None}})


def test_validation_fits_preserves_reference_arrays(monkeypatch):
    arrays = dict(x_data=np.arange(8).reshape(4, 2),
                  pk_m=np.ones((4, 3)), fk_m=np.full((4, 3), 2.),
                  Z_ARRAY=np.arange(4), REF_fk_m=np.ones((3, 4)))
    monkeypatch.setattr(hist.io, 'FitsFile', lambda path:
                        SimpleNamespace(get_data=arrays.__getitem__))
    view = hist.ValidationFits('sample.fits', np.array([1, 3]), {'pk_m', 'fk_m'})
    for key in ('x_data', 'pk_m', 'fk_m'):
        np.testing.assert_array_equal(view.get_data(key), arrays[key][[1, 3]])
    for key in ('Z_ARRAY', 'REF_fk_m'):
        np.testing.assert_array_equal(view.get_data(key), arrays[key])
