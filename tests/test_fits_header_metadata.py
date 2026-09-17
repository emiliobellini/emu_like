"""Provenance survives generation, loading, and legacy header migration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from emu_like import io
from emu_like.datasets import DataCollection
from test_sampling_redshift_coverage import CoverageClass, CoverageTests


def test_sampling_keeps_input_and_records_effective_arguments(tmp_path):
    path = tmp_path / 'sample.fits'
    runtime = SimpleNamespace(HiClass=CoverageClass,
                              CosmoComputationError=RuntimeError,
                              CosmoSevereError=RuntimeError)
    args = {'output': 'mPk'}
    with patch('emu_like.y_models.hiclassy', runtime):
        DataCollection().sample(
            params={'z_pk': {'prior': {'min': 0., 'max': 1.}}},
            x_name='grid', x_args={'n_samples': 2},
            y_name='class_spectra', y_args=args,
            y_outputs={'pk_weyl': dict(k_min=.01, k_max=1., k_num=5,
                                       k_space='log', ratio=True)},
            output=str(path), prepare_only=True)
    file = io.FitsFile(str(path))
    assert args == {'output': 'mPk'}
    assert file.get_header(0)['y_model']['args'] == args
    assert file.get_header('X_DATA')['parameters'] == {
        'z_pk': {'column': 0, 'prior': {'min': 0., 'max': 1.}}}
    header = file.get_header('PK_WEYL')
    assert header['class_parameters']['output'] == 'mPk, wPk'
    assert 'z_pk' not in header['class_parameters']
    assert header['stored_units'] == 'dimensionless'


def test_load_restores_reference_parameters_and_grids(tmp_path):
    model = CoverageTests().model()
    model.ref_params['h'] = .7123
    path = tmp_path / 'ref.fits'
    io.FitsFile(str(path)).write(header={'test': True})
    model.save(str(path))
    loaded = CoverageTests().model()
    loaded.load(str(path))
    assert loaded.ref_params == model.ref_params
    for sp in loaded.spectra:
        np.testing.assert_array_equal(sp.z_array, loaded.z_array)
    second = tmp_path / 'second.fits'
    io.FitsFile(str(second)).write(header={'test': True})
    loaded.save(str(second))
    assert io.FitsFile(str(second)).get_header('REF_PK_M') == model.ref_params


def test_selected_spectrum_retains_full_calculation_coverage():
    model = CoverageTests().model()
    selected = model['pk_m']
    assert selected.get_storage_headers()[0]['class_metadata']['z_max_pk']['step'] == .001


def test_conflicting_reference_headers_are_rejected(tmp_path):
    import pytest
    model = CoverageTests().model()
    path = tmp_path / 'conflicting.fits'
    file = io.FitsFile(str(path))
    file.write(header={'test': True})
    model.save(str(path))
    file.update('REF_PK_CB', header=model.ref_params | {'h': .9})
    with pytest.raises(ValueError, match='headers disagree'):
        model.load(str(path))


def test_migration_preserves_arrays_primary_references_and_checksums(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'update_headers', Path(__file__).parents[1] /
        'scripts/check_data/update_fits_headers.py')
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    model = CoverageTests().model()
    path = tmp_path / 'legacy.fits'
    file = io.FitsFile(str(path))
    file.write(header={'params': model.params, 'y_model': {
        'name': 'class_spectra', 'args': {'output': 'mPk'},
        'outputs': model.outputs}})
    file.write(name='X_DATA', data=np.array([[0.], [1.]]))
    model.save(str(path))
    for sp, values in zip(model.spectra, model.y):
        file.write(name=sp.name, data=values, header=sp.get_header())
    with fits.open(path, mode='update') as hdus:
        for hdu in hdus:
            hdu.add_checksum()
    before = path.read_bytes()
    assert migration.migrate(path)
    assert path.read_bytes() == before
    migration.migrate(path, apply=True)
    backup = path.with_name(path.name + '.before_header_update.bak')
    assert backup.read_bytes() == before
    with fits.open(path) as after, fits.open(backup) as old:
        assert after[0].header == old[0].header
        for a, b in zip(after, old):
            if a.data is not None:
                np.testing.assert_array_equal(a.data, b.data)
            if a.name.startswith('REF_'):
                assert a.header == b.header
            assert a.verify_checksum() == 1
    assert not migration.migrate(path, apply=True)
    assert file.get_header('PK_WEYL')['class_metadata']['provenance'].startswith('inferred')
