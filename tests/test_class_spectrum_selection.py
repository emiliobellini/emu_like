"""Use computed CLASS state for cb fallback and nonlinear table selection."""
import unittest

import numpy as np

from emu_like.spectra import (
    ColdBaryonPk, ColdBaryonGrowthRate, WeylPk, WeylGrowthRate,
    CosmoSevereError,
)
from test_reference_pk_sampling import FakeClass


PARAMS = dict(k_min=.02, k_max=.2, k_num=5, k_space='log')


class BrokenCb(FakeClass):
    def pk_cb(self, k, z):
        raise CosmoSevereError('cb evaluation failed')


class WeylClass(FakeClass):
    def pk_weyl(self, k, z):
        if self.nonlinear_method:
            raise CosmoSevereError('Nonlinear Weyl unsupported')
        return k**.96 * (1+z)**2

    def get_Weyl_pk_and_k_and_z(self, **kwargs):
        raise AssertionError('Weyl must use its evaluator')


class SelectionTests(unittest.TestCase):
    def test_cb_errors_propagate_when_non_cold_matter_is_present(self):
        for cls in (ColdBaryonPk, ColdBaryonGrowthRate):
            for z in (None, .5):
                with self.subTest(cls=cls.__name__, z=z):
                    with self.assertRaisesRegex(
                            CosmoSevereError, 'cb evaluation failed'):
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
                        if method:
                            with self.assertRaisesRegex(
                                    CosmoSevereError, 'Nonlinear Weyl'):
                                cls('weyl', PARAMS).get(c, z)
                        else:
                            sp = cls('weyl', PARAMS)
                            values = sp.get(c, z)
                            self.assertTrue(np.all(np.isfinite(values)))
                            if cls is WeylGrowthRate:
                                np.testing.assert_allclose(
                                    values, -1., atol=1e-9)
                            else:
                                zz = sp.z_array if z is None else z
                                expected = (sp.k_range*c.h())**.96
                                if z is None:
                                    expected = expected[:, None]
                                expected = expected*(1+zz)**2*c.h()**3
                                np.testing.assert_allclose(values, expected)


if __name__ == '__main__':
    unittest.main()
