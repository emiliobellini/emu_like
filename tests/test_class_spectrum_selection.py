"""Use computed CLASS state for cb fallback and nonlinear table selection."""
import unittest

import numpy as np

from emu_like.spectra import (
    ColdBaryonPk, ColdBaryonGrowthRate, WeylPk, WeylGrowthRate,
    ClassySevereError,
)
from test_reference_pk_sampling import FakeClass


PARAMS = dict(k_min=.02, k_max=.2, k_num=5, k_space='log')


class BrokenCb(FakeClass):
    def pk_cb(self, k, z):
        raise ClassySevereError('cb evaluation failed')


class WeylClass(FakeClass):
    def get_Weyl_pk_and_k_and_z(self, nonlinear, h_units):
        assert nonlinear == bool(self.nonlinear_method)
        k = np.array([.001, .01, .1, 1.])
        z = np.array([2., 1., .5, 0.])
        return k[:, None] * (1+z[None, :])**2, k, z


class SelectionTests(unittest.TestCase):
    def test_cb_errors_propagate_when_non_cold_matter_is_present(self):
        for cls in (ColdBaryonPk, ColdBaryonGrowthRate):
            for z in (None, .5):
                with self.subTest(cls=cls.__name__, z=z):
                    with self.assertRaisesRegex(
                            ClassySevereError, 'cb evaluation failed'):
                        cls('cb', PARAMS).get(BrokenCb(has_cb=True), z)

    def test_cb_uses_total_matter_only_without_non_cold_matter(self):
        c = BrokenCb(has_cb=False)
        sp = ColdBaryonPk('pk_cb', PARAMS)
        values = sp.get(c, .5)
        expected = np.array([c.pk(k*c.h(), .5) for k in sp.k_range])*c.h()**3
        np.testing.assert_allclose(values, expected)
        np.testing.assert_allclose(sp.get(c)[:, 1], expected)
        growth = ColdBaryonGrowthRate('fk_cb', PARAMS).get(c, .5)
        np.testing.assert_allclose(growth, 1., atol=1e-5)

    def test_weyl_uses_active_method_including_explicit_none(self):
        for cls in (WeylPk, WeylGrowthRate):
            for method in (0, 1, 2):
                for z in (None, .5):
                    with self.subTest(cls=cls.__name__, method=method, z=z):
                        c = WeylClass()
                        c.nonlinear = method
                        c.pars = {'non_linear': 'none'} if method == 0 else {}
                        values = cls('weyl', PARAMS).get(c, z)
                        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == '__main__':
    unittest.main()
