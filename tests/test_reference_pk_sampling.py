"""Reference P(k,z) must use the same CLASS evaluator as sampled P(k)."""
import unittest

import numpy as np

from emu_like.spectra import MatterPk, ColdBaryonPk, ClassySevereError


class FakeClass:
    def __init__(self, nonlinear=False, has_cb=True):
        self.pars = {'non_linear': 'halofit'} if nonlinear else {}
        self.nonlinear = nonlinear
        self.has_cb = has_cb

    def h(self):
        return 0.7

    def get_pk_and_k_and_z(self, **kwargs):
        # Deliberately unsuitable for extrapolation; only z should be used.
        return np.ones((4, 4)), np.array([0.01, 0.1, 0.2, 1.]), np.array([2., 1., .5, 0.])

    def pk(self, k, z):
        if k > 1.:
            raise ClassySevereError('k out of bounds')
        return (1 + self.nonlinear * k**2) * k**0.96 / (1 + z)**2

    def pk_cb(self, k, z):
        if not self.has_cb:
            raise ClassySevereError('P_cb not computed')
        return 1.2 * self.pk(k, z)


class ReferencePowerTests(unittest.TestCase):
    def test_reference_matches_samples_including_extrapolation_and_units(self):
        params = dict(k_min=1e-5, k_max=1., k_num=7, k_space='log')
        for cls, species in ((MatterPk, 'm'), (ColdBaryonPk, 'cb')):
            for nonlinear in (False, True):
                for has_cb in (False, True):
                    with self.subTest(species=species, nonlinear=nonlinear, has_cb=has_cb):
                        cosmo = FakeClass(nonlinear, has_cb)
                        sp = cls('pk_' + species, params)
                        table = sp.get(cosmo)
                        self.assertEqual(table.shape, (7, 4))
                        np.testing.assert_array_equal(sp.z_array, [0., .5, 1., 2.])
                        for j, z in enumerate(sp.z_array):
                            np.testing.assert_allclose(table[:, j], sp.get(cosmo, z))
                        physical_k = sp.k_range * cosmo.h()
                        expected = (1 + nonlinear * physical_k**2) * physical_k**0.96 * cosmo.h()**3
                        if species == 'cb' and has_cb:
                            expected *= 1.2
                        np.testing.assert_allclose(table[:, 0], expected)
                        self.assertTrue(np.all(table > 0))

    def test_high_k_errors_are_not_spline_extrapolated(self):
        for cls, species in ((MatterPk, 'm'), (ColdBaryonPk, 'cb')):
            sp = cls('pk_' + species, dict(
                k_min=.1, k_max=2., k_num=4, k_space='log'))
            with self.assertRaisesRegex(ClassySevereError, 'out of bounds'):
                sp.get(FakeClass())


if __name__ == '__main__':
    unittest.main()
