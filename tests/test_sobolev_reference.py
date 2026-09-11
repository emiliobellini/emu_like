"""Physical reference growth must not depend on stored fk normalization."""
import unittest
from unittest.mock import patch

import numpy as np

from emu_like.datasets import Dataset, SobolevDataset


class ReferenceGrowthTests(unittest.TestCase):
    def test_analytic_reference_and_constant_normalization(self):
        z = np.linspace(0., 3., 31)
        pk = np.array([2. + z + z**2, 4. + 2*z])[None]
        expected = -.5 * (1 + z) * np.array([1 + 2*z, 2*np.ones_like(z)])[None] / pk
        np.testing.assert_allclose(
            SobolevDataset.growth_from_reference_pk(pk, z), expected,
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            SobolevDataset.growth_from_reference_pk(np.ones_like(pk), z), 0.,
            atol=1e-12)

    def test_loader_is_invariant_to_fk_normalization(self):
        z = np.linspace(0., 3., 31)
        sample_z = np.array([.15, 1.25, 2.7])
        physical = np.array([[.6, .7], [.8, .9], [.95, 1.]])
        pk = np.array([2. + z + z**2, 4. + 2*z])[None]
        references = []
        for normalized in (False, True):
            norm = np.array([1 + .2*z, 2 + .1*z]) if normalized else np.ones((2, len(z)))
            sample_norm = np.array([1 + .2*sample_z, 2 + .1*sample_z]).T if normalized else 1.
            arrays = {'fk_m': physical / sample_norm, 'REF_PK_M': pk,
                      'REF_FK_M': norm[None], 'Z_ARRAY': z}

            def load_primary(instance, **kwargs):
                instance.x = np.column_stack([np.ones(3), sample_z])
                instance.x_names = ['a', 'z_pk']
                instance.y = np.ones((3, 2))
                instance.n_y = 2

            class FakeFits:
                def __init__(self, path):
                    pass

                def get_keys(self):
                    return list(arrays)

                def get_data(self, key):
                    return arrays[key]

            with patch.object(Dataset, 'load', load_primary), patch(
                    'emu_like.datasets.io.FitsFile', FakeFits):
                data = SobolevDataset.__new__(SobolevDataset)
                data.load('unused', name='pk_m')
            np.testing.assert_allclose(data.y_growth, physical, rtol=1e-12)
            references.append(data.reference_growth)
        np.testing.assert_array_equal(*references)

    def test_invalid_reference_is_rejected(self):
        for pk, z in [(np.ones((1, 3)), [0, 1, 1]),
                      (np.zeros((1, 3)), [0, 1, 2]),
                      (np.ones((1, 2)), [0, 1, 2])]:
            with self.assertRaises(ValueError):
                SobolevDataset.growth_from_reference_pk(pk, z)


if __name__ == '__main__':
    unittest.main()
