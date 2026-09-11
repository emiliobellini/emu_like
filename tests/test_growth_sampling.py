"""Growth targets differentiate the configured CLASS power evaluator."""
import unittest

import numpy as np

from emu_like.spectra import MatterGrowthRate, ColdBaryonGrowthRate
from test_reference_pk_sampling import FakeClass


class GrowthClass(FakeClass):
    def pk(self, k, z):
        if not 0 <= z <= 2:
            raise AssertionError('Stencil outside computed coverage')
        # Redshift and k dependence expose ordering and unit-conversion bugs.
        return k**0.96 * np.exp(-(1 + k + 0.2*self.nonlinear)*z)

    def pk_cb(self, k, z):
        return self.pk(k, z) * np.exp(-0.3*z)


class GrowthSamplingTests(unittest.TestCase):
    def test_species_boundaries_order_and_reference_grid(self):
        params = dict(k_min=1e-5, k_max=1., k_num=7, k_space='log')
        for cls, species in ((MatterGrowthRate, 'm'), (ColdBaryonGrowthRate, 'cb')):
            for nonlinear in (False, True):
                with self.subTest(species=species, nonlinear=nonlinear):
                    c = GrowthClass(nonlinear)
                    sp = cls('fk_' + species, params)
                    z = np.array([1., 0., 2., .0005, 1.9995, 1.])
                    rate = 1 + sp.k_range*c.h() + 0.2*nonlinear + 0.3*(species == 'cb')
                    expected = 0.5*rate[:, None]*(1+z)
                    np.testing.assert_allclose(sp.get(c, z), expected, atol=8e-6, rtol=0)
                    table = sp.get(c)
                    self.assertEqual(table.shape, (7, 4))
                    np.testing.assert_array_equal(sp.z_array, [0., .5, 1., 2.])
                    for j, zi in enumerate(sp.z_array):
                        scalar = sp.get(c, float(zi))
                        self.assertEqual(scalar.shape, (7,))
                        np.testing.assert_allclose(table[:, j], scalar)
                    for bad_z in (-.1, 2.1, np.nan):
                        with self.assertRaisesRegex(ValueError, 'inside the CLASS table'):
                            sp.get(c, bad_z)

    def test_low_k_growth_is_constant_for_class_style_extrapolation(self):
        params = dict(k_min=1e-5, k_max=1e-4, k_num=9, k_space='log')
        for cls in (MatterGrowthRate, ColdBaryonGrowthRate):
            sp = cls('fk', params)
            values = sp.get(FakeClass(), [0., .5, 2.])
            np.testing.assert_allclose(values, 1., atol=1e-5, rtol=0)
            np.testing.assert_allclose(values, np.broadcast_to(values[:1], values.shape), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
