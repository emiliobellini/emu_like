"""Normalized sampling cannot extrapolate its saved reference spectra."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import test_sampling_redshift_coverage as coverage


class NormalizationTests(unittest.TestCase):
    def model(self):
        m = coverage.CoverageTests().model()
        for sp in m.spectra:
            sp.ratio = True
        # Smooth exact cubic references make normalization independently known.
        for i, ref in enumerate(m.y_ref):
            m.y_ref[i] = np.broadcast_to(
                (2 + m.z_array**2)[None, None, :], ref.shape).copy()
        m._validate_reference_tables()
        return m

    def test_endpoints_and_interior_normalization(self):
        m = self.model()
        for z in (0., .4, 1.):
            result = m.evaluate([z], 0)
            for i, sp in enumerate(m.spectra):
                expected = sp.get(m.cosmo, z)/(2 + z**2)
                np.testing.assert_allclose(result[i][0], expected, rtol=1e-12)

    def test_outside_reference_fails_before_compute_or_row_write(self):
        m = self.model()
        original = copy.deepcopy(m.y)
        for z in (-1e-10, 1+1e-10, np.nan):
            with self.assertRaisesRegex(ValueError, 'Reference for .* covers'):
                m.evaluate([z], 0)
        self.assertEqual(m.cosmo.computations, [])
        for before, after in zip(original, m.y):
            np.testing.assert_array_equal(before, after)

    def test_zero_normalizer_is_rejected(self):
        m = self.model()
        m.y_ref[0][:] = 0.
        with self.assertRaisesRegex(ValueError, 'zero or nonfinite'):
            m.evaluate([.4], 0)
        self.assertEqual(m.cosmo.computations, [])

    def test_loaded_references_are_validated(self):
        for defect in ('unordered', 'shape', 'nonfinite'):
            with self.subTest(defect=defect):
                m = self.model()
                arrays = {'z_array': m.z_array.copy()}
                for i, sp in enumerate(m.spectra):
                    arrays['ref_' + sp.name] = m.y_ref[i].copy()
                    arrays['k_range_' + sp.name] = sp.k_range
                if defect == 'unordered':
                    arrays['z_array'] = arrays['z_array'][::-1]
                elif defect == 'shape':
                    arrays['ref_pk_m'] = arrays['ref_pk_m'][..., :-1]
                else:
                    arrays['ref_pk_m'][0, 0, 0] = np.nan
                with patch('emu_like.y_models.io.FitsFile') as fits:
                    fits.return_value.get_data.side_effect = arrays.__getitem__
                    with self.assertRaises(ValueError):
                        m.load('reference.fits')

    def test_unnormalized_outputs_do_not_require_reference_coverage(self):
        m = coverage.CoverageTests().model()
        result = m.evaluate([2.], 0)
        self.assertTrue(all(np.all(np.isfinite(y)) for y in result))


if __name__ == '__main__':
    unittest.main()
