"""Sampling and reference outputs reserve their derivative coverage."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from emu_like.y_models import ClassSpectra
from test_reference_pk_sampling import FakeClass


class CoverageClass(FakeClass):
    instances = []

    def __init__(self):
        super().__init__()
        self.calls = []
        self.computations = []
        self.instances.append(self)

    def set(self, params):
        self.pars = params.copy()

    def compute(self):
        self.computations.append(self.pars.copy())

    def get_pk_and_k_and_z(self, **kwargs):
        z = np.linspace(self.pars['z_max_pk'], 0., 9)
        return np.ones((4, len(z))), np.array([.01, .1, .2, 1.]), z

    def pk(self, k, z):
        assert 0 <= z <= self.pars['z_max_pk']
        self.calls.append(z)
        return super().pk(k, z)


class CoverageTests(unittest.TestCase):
    def model(self, configured=None, growth=True):
        runtime = SimpleNamespace(Class=CoverageClass,
                                  CosmoComputationError=RuntimeError,
                                  CosmoSevereError=RuntimeError)
        names = ['pk_m', 'pk_cb'] + (['fk_m', 'fk_cb'] if growth else [])
        outputs = {name: dict(k_min=1e-5, k_max=1., k_num=5,
                              k_space='log', ratio=False) for name in names}
        args = {} if configured is None else {'z_max_pk': configured}
        with patch('emu_like.y_models.classy', runtime):
            return ClassSpectra(name='class', params={
                'z_pk': {'prior': {'min': 0., 'max': 1.}}},
                n_samples=2, outputs=outputs, **args)

    def test_sample_coverage_preserves_configuration_and_has_no_row_history(self):
        for limit in (None, 3.):
            m = self.model(limit)
            m.evaluate([1.], 0)
            m.evaluate([.2], 1)
            expected = [1.001, .201] if limit is None else [3., 3.]
            np.testing.assert_allclose(
                [p['z_max_pk'] for p in m.cosmo.computations], expected)
            self.assertIn(1.001, m.cosmo.calls)
            self.assertIn(.201, m.cosmo.calls)

    def test_reference_outputs_exclude_stencil_padding_and_share_grid(self):
        m = self.model()
        self.assertAlmostEqual(m.ref_params['z_max_pk'], 1.001)
        self.assertEqual(m.z_array[-1], 1.)
        for sp, ref in zip(m.spectra, m.y_ref):
            np.testing.assert_array_equal(sp.z_array, m.z_array)
            self.assertEqual(ref.shape, (1, 5, len(m.z_array)))
        reference_cosmo = CoverageClass.instances[-1]
        self.assertIn(1.001, reference_cosmo.calls)
        # The join coverage check must not mistake padding for output range.
        z, _ = ClassSpectra._join_references([m, m], m.ref_params)
        np.testing.assert_array_equal(z, m.z_array)

    def test_no_growth_needs_no_derivative_padding(self):
        m = self.model(growth=False)
        self.assertEqual(m.ref_params['z_max_pk'], 1.)
        m.evaluate([1.], 0)
        self.assertEqual(m.cosmo.pars['z_max_pk'], 1.)

    def test_origin_reserves_forward_stencil(self):
        m = self.model()
        m.evaluate([0.], 0)
        self.assertIn(.002, m.cosmo.calls)
        self.assertGreaterEqual(m.cosmo.pars['z_max_pk'], .002)


if __name__ == '__main__':
    unittest.main()
