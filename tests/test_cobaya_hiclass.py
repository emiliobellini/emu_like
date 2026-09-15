"""The Cobaya adapter must use HiClass, including its error types."""
import numpy as np
import pytest

hiclassy = pytest.importorskip('hiclassy')
cobaya = pytest.importorskip('cobaya.model')


@pytest.mark.parametrize('extra_args', [
    {'non_linear': None}, {'non linear': False}, {'non_linear': 'none'},
])
def test_hiclass_background(extra_args, monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    name = 'emu_like.cobaya_hiclass.HiClassTheory'
    info = {
        'params': {'H0': 67., 'omega_b': .0224, 'omega_cdm': .12},
        'theory': {name: {'extra_args': extra_args}},
        'likelihood': {'test': {
            'external': lambda _self: 0.,
            'requires': {'Hubble': {'z': [0.]}},
        }},
    }
    with cobaya.get_model(info) as model:
        theory = model.theory[name]
        assert isinstance(theory.classy, hiclassy.HiClass)
        assert theory.classy_module is hiclassy
        assert theory.extra_args['non_linear'] == 'none'
        assert model.logposterior({}).logpost == 0.
        np.testing.assert_allclose(model.provider.get_Hubble(0.), 67.)
