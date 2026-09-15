"""Range persistence, overlap, holes, and failure recovery."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from emu_like.datasets import DataCollection
from emu_like.range_sampling import sample_range, merge_ranges, lock, MASK
from emu_like.y_models import Linear1D


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'main.fits'
        self.params = dict(params={'x': {'prior': {'min': 0., 'max': 9.}}},
                           x_name='grid', x_args={'n_samples': 10},
                           y_name='linear_1d', y_args={'a': 2., 'b': -1.})
        DataCollection().sample(
            **self.params, output=str(self.path), prepare_only=True)

    def check(self, rows):
        with fits.open(self.path) as hdus:
            done = hdus[MASK].data.astype(bool)
            np.testing.assert_array_equal(np.flatnonzero(done), rows)
            np.testing.assert_allclose(hdus['y_data'].data[done, 0],
                                       2 * hdus['x_data'].data[done, 0] - 1)
            self.assertTrue(np.isnan(hdus['y_data'].data[~done]).all())

    def test_out_of_order_and_resume_holes(self):
        part = sample_range(self.path, 5, 10, save_interval=2)
        merge_ranges(self.path, [part])
        self.assertFalse(Path(part).exists())
        self.check(np.arange(5, 10))
        data = DataCollection().load(str(self.path))
        self.assertEqual(data.counter_samples, 5)
        data.resume(str(self.path), save_interval=2)
        self.check(np.arange(10))

    def test_interruption_and_restart(self):
        original = Linear1D.evaluate

        def fail(model, x, idx):
            if idx == 7:
                raise RuntimeError('node failed')
            return original(model, x, idx)
        with patch.object(Linear1D, 'evaluate', fail):
            with self.assertRaisesRegex(RuntimeError, 'node failed'):
                sample_range(self.path, 5, 10, save_interval=2)
        part = next(self.path.parent.glob('*.rows-*.fits'))
        with fits.open(part) as hdus:
            np.testing.assert_array_equal(hdus['ROW_IDS'].data, [5, 6])
        called = []

        def track(model, x, idx):
            called.append(idx)
            return original(model, x, idx)
        with patch.object(Linear1D, 'evaluate', track):
            sample_range(self.path, 5, 10)
        self.assertEqual(called, [7, 8, 9])
        merge_ranges(self.path, [part])
        self.check(np.arange(5, 10))

    def test_overlap_last_wins_and_atomic_failure(self):
        first = sample_range(self.path, 0, 6)
        second = sample_range(self.path, 4, 10)
        with fits.open(first, mode='update') as hdus:
            hdus['y_data'].data[:] = 123
        before = self.path.read_bytes()
        with patch('emu_like.range_sampling.os.replace',
                   side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                merge_ranges(self.path, [first, second])
        self.assertEqual(before, self.path.read_bytes())
        self.assertTrue(Path(first).exists())
        merge_ranges(self.path, [first, second])
        with fits.open(self.path) as hdus:
            np.testing.assert_array_equal(hdus['y_data'].data[:4], 123)
            np.testing.assert_allclose(hdus['y_data'].data[4:, 0],
                                       2 * hdus['x_data'].data[4:, 0] - 1)

    def test_lock_and_identity_rejection(self):
        part = sample_range(self.path, 0, 3)
        with lock(part):
            with self.assertRaises(RuntimeError):
                merge_ranges(self.path, [part])
        with fits.open(self.path, mode='update') as hdus:
            hdus['x_data'].data[0, 0] = 42
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            merge_ranges(self.path, [part])
        self.assertTrue(Path(part).exists())

    def test_multi_output_nan_and_invalid_row_ids(self):
        class TwoOutputs(Linear1D):
            def __init__(self, name, params, outputs, count, **kwargs):
                super().__init__(name, params, count, **kwargs)
                self.y_keys = ['first', 'second']
                self.y = [np.zeros((count, 1)), np.zeros((count, 2))]
                self.n_y = [1, 2]

            def evaluate(self, x, idx):
                return [np.array([[idx]]), np.array([[idx, np.nan]])]

        other = self.path.parent / 'multi.fits'
        with patch('emu_like.y_models.YModel.choose_one',
                   side_effect=TwoOutputs):
            DataCollection().sample(
                **self.params, output=str(other), prepare_only=True)
            part = sample_range(other, 3, 7, save_interval=1)
            with fits.open(part, mode='update') as hdus:
                hdus['ROW_IDS'].data[1] = 3
            before = other.read_bytes()
            with self.assertRaisesRegex(ValueError, 'duplicate row IDs'):
                merge_ranges(other, [part])
            self.assertEqual(before, other.read_bytes())
            with fits.open(part, mode='update') as hdus:
                hdus['ROW_IDS'].data[1] = 4
            merge_ranges(other, [part])
            with fits.open(other) as hdus:
                np.testing.assert_array_equal(
                    np.flatnonzero(hdus[MASK].data), [3, 4, 5, 6])
                np.testing.assert_array_equal(
                    hdus['first'].data[3:7, 0], [3, 4, 5, 6])
                np.testing.assert_array_equal(
                    hdus['second'].data[3:7, 0], [3, 4, 5, 6])
                self.assertTrue(np.isnan(hdus['second'].data[:, 1]).all())

    def test_legacy_prefix_preserved(self):
        # An ordinary sampler still writes a prefix without a mask.
        other = self.path.parent / 'legacy.fits'
        DataCollection().sample(
            **self.params, output=str(other), timeout=0, save_interval=1)
        part = sample_range(other, 7, 10)
        merge_ranges(other, [part])
        with fits.open(other) as hdus:
            np.testing.assert_array_equal(
                np.flatnonzero(hdus[MASK].data), [0, 7, 8, 9])


if __name__ == '__main__':
    unittest.main()
