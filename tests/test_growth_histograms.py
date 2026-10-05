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
    hist.plot_observable(tmp_path, 'fk_m', [dict(
        range='test', groups=groups, grid=k, input_rows=4)],
        [.01, .05, .1, 1.], None, None)
    assert {p.name for p in tmp_path.glob('*.png')} == {
        'fk_accuracy.png', 'fk_errors_vs_k.png'}
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


def test_relative_accuracy_uses_percent_rms_and_excludes_undefined_rows():
    residual = np.array([[.03, .04], [.3, .4], [np.nan, 1.]])
    row = hist.relative_accuracy_row('thin', 'fk_from_pk', residual, 4, [.01, .1, 1.])
    assert row[:5] == ['thin', 'fk_from_pk', 4, 2, 2]
    assert row[-3:] == ['100.000%', '50.000%', '0.000%']
    np.testing.assert_allclose(float(row[5]), np.mean([np.sqrt(.00125), np.sqrt(.125)]), rtol=1e-5)
    assert hist.relative_accuracy_row(
        'thin', 'fk_from_pk', residual[:0], 2, [.1])[-1] == 'N/A'


@pytest.mark.parametrize('spectrum,sobolev', [
    ('pk_m', False), ('pk_m', True), ('fk_m', False),
    ('cl_TT_lensed', False), ('cl_TE_lensed', False)])
@pytest.mark.parametrize('growth_histograms', [False, True])
def test_compact_summary_and_output_files(
        tmp_path, monkeypatch, spectrum, sobolev, growth_histograms):
    import yaml
    params = dict(datasets=dict(name=spectrum, paths=['sample_thin.fits']),
                  emulator=dict(name='sobolev_ffnn_emu' if sobolev else 'ffnn_emu'))
    (tmp_path / 'params.yaml').write_text(yaml.safe_dump(params))
    grid = np.array([.1, 1.]) if not spectrum.startswith('cl_') else np.array([2., 100.])
    arrays = dict(x_data=np.arange(4.)[:, None],
                  pk_m=np.ones((4, 2)), fk_m=np.ones((4, 2)),
                  Z_ARRAY=np.arange(4.))
    arrays[spectrum] = np.ones((4, 2))
    for name in {spectrum, 'fk_m'}:
        arrays['REF_' + name] = (np.ones((1, 2)) if name.startswith('cl_')
                                 else np.ones((2, 4)))
        arrays[('ELL_RANGE_' if name.startswith('cl_') else 'K_RANGE_') + name] = grid
    monkeypatch.setattr(hist, 'validation_rows', lambda params: [np.array([1, 3])])
    monkeypatch.setattr(hist.io, 'FitsFile', lambda path:
                        SimpleNamespace(get_data=arrays.__getitem__))

    class FakeEmuData(hist.EmuData):
        def __init__(self, root):
            self.is_sobolev = sobolev
            self.emu = SimpleNamespace(
                epochs=[1, 2], loss=[.2, .1], val_loss=[.3, .2],
                learning_rate=[.001, .001], x_names=['z_pk'],
                y_model=SimpleNamespace(k_ranges=[grid], ell_ranges=[grid],
                    z_array=arrays['Z_ARRAY'], y_ref=[arrays['REF_' + spectrum]]))

        def get_y_emu(self, x, **kwargs):
            np.testing.assert_array_equal(x[:, 0], [1, 3])
            return np.ones((2, 2)) + np.array([[.0002], [.002]])

    monkeypatch.setattr(hist, 'EmuData', FakeEmuData)
    calls = []

    def residuals(emudata, dataset, spectrum, *args):
        calls.append(spectrum)
        np.testing.assert_array_equal(dataset.get_data('x_data')[:, 0], [1, 3])
        relative = {'fk_from_pk': np.array([[.02, .02], [.2, .2]])}
        groups = {'accuracy_relative_percent': relative,
                  'accuracy_absolute': {'fk_from_pk': relative['fk_from_pk']/100}}
        if sobolev:
            relative['eval_fk'] = np.array([[3., 3.], [3., 3.]])
            groups['consistency_absolute'] = {'eval_fk - fk_from_pk': np.ones((2, 2))*.03}
        return groups, {}

    monkeypatch.setattr(hist, 'growth_residuals', residuals)
    observations = hist.show_summary(tmp_path, growth_histograms=growth_histograms)
    summary = (tmp_path / 'summary_table.txt').read_text()
    assert f'{spectrum} (validation; relative RMS accuracy)' in summary
    assert 'Median RMS [%]' in summary and 'P95 RMS [%]' in summary
    assert '50.000%' in summary
    assert 'Training history' in summary and 'Diagnostic timing' in summary
    family = spectrum.split('_')[0]
    expected = {f'{family}_accuracy.png',
                f'{family}_errors_vs_{"ell" if family == "cl" else "k"}.png'}
    if family == 'pk':
        assert 'fk_m (validation; relative RMS accuracy)' in summary
        assert calls == ['fk_m']
        assert ('Sobolev consistency' in summary) == sobolev
        if growth_histograms:
            expected.update(['fk_accuracy.png', 'fk_errors_vs_k.png'])
    else:
        assert calls == []
        assert list(observations) == [spectrum]
    assert {p.name for p in tmp_path.glob('*.png')} == expected
    assert {p.name for p in tmp_path.iterdir()} == expected | {'params.yaml', 'summary_table.txt'}


@pytest.mark.parametrize('spectrum', ['pk_m', 'fk_m', 'cl_TE_lensed'])
def test_direct_residuals_restore_separate_references(spectrum):
    # Different normalizations represent identical physical spectra.
    is_cl = spectrum.startswith('cl_')
    data = {'x_data': np.array([[.5], [1.]]), spectrum: np.array([[2., -2.], [2., -2.]]),
            'REF_' + spectrum: np.ones((1, 2)) if is_cl else np.ones((2, 2)),
            'Z_ARRAY': np.array([0., 2.])}
    model = SimpleNamespace(x_names=['z_pk'], y_model=SimpleNamespace(
        y_ref=[np.full((1, 2) if is_cl else (2, 2), 2.)], z_array=data['Z_ARRAY']))
    emudata = SimpleNamespace(emu=model, get_y_emu=lambda x, **kw: np.array([[1., -1.], [1., -1.]]))
    result = hist.direct_residuals(emudata, SimpleNamespace(get_data=data.__getitem__), spectrum, 1e-6)
    np.testing.assert_allclose(result['accuracy_relative_percent']['emulator'], 0.)
    data[spectrum][0, 0] = 0.
    result = hist.direct_residuals(emudata, SimpleNamespace(get_data=data.__getitem__), spectrum, 1e-6)
    assert np.isnan(result['accuracy_relative_percent']['emulator'][0, 0])
    assert len(hist.complete_errors(result['accuracy_relative_percent']['emulator'])[1]) == 1


def test_compact_plots_with_zero_and_undefined_errors(tmp_path):
    groups = {'accuracy_relative_percent': {'emulator': np.zeros((1, 2))}}
    records = [dict(range='zero', grid=np.array([2, 3]), groups=groups, input_rows=1),
               dict(range='empty', grid=np.array([2, 3]), input_rows=2,
                    groups={'accuracy_relative_percent': {'emulator': np.full((2, 2), np.nan)}})]
    hist.plot_observable(tmp_path, 'cl_TE_lensed', records, [.01, .05, .1, 1.], None, None)
    assert len(list(tmp_path.glob('*.png'))) == 2
