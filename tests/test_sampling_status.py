"""Progress reporting counts persisted row IDs without loading spectra."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from emu_like.io import FitsFile

spec = importlib.util.spec_from_file_location(
    'sampling_status', Path(__file__).resolve().parents[1] /
    'scripts/check_data/check_sampling_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'sample.fits'
        file = FitsFile(str(self.path))
        file.write(header={'y_model': {'outputs': {
            'first': {'ratio': False}, 'second': {'ratio': False}}}})
        file.write(name='x_data', data=np.zeros((12, 1)))
        for key in ('first', 'second'):
            file.write(name=key, data=np.full((2, 1), np.nan))

    def part(self, start, stop, rows):
        path = self.path.with_name(f'sample.rows-{start:09d}-{stop:09d}.fits')
        fits.HDUList([
            fits.PrimaryHDU(header=fits.Header(
                {'START': start, 'STOP': stop})),
            fits.ImageHDU(np.array(rows, dtype=np.int64), name='ROW_IDS'),
            fits.ImageHDU(np.zeros((len(rows), 1)), name='first'),
            fits.ImageHDU(np.zeros((len(rows), 1)), name='second'),
        ]).writeto(path)
        return path

    def test_overlap_and_legacy_nan_rows(self):
        self.part(0, 6, [1, 4, 5])
        self.part(5, 9, [5, 6])
        report = status.inspect_dataset(self.path)
        self.assertEqual(
            (report['main'], report['unmerged'], report['saved'],
             report['missing']), (2, 3, 5, 7))
        self.assertEqual(report['ranges'], [(0, 6, 3), (5, 9, 2)])
        self.assertFalse(report['errors'])

    def test_mask_and_invalid_checkpoint(self):
        with fits.open(self.path, mode='update') as hdus:
            for key in ('first', 'second'):
                hdus[key].data = np.zeros((12, 1))
            mask = np.zeros(12, dtype=np.uint8)
            mask[[3, 10]] = 1
            hdus.append(fits.ImageHDU(mask, name='SAMPLE_DONE'))
        self.part(0, 6, [2, 3])
        self.part(7, 9, [7, 7])
        report = status.inspect_dataset(self.path)
        self.assertEqual(
            (report['main'], report['saved'], report['missing']), (2, 3, 9))
        self.assertEqual(len(report['errors']), 1)

    def test_yaml_log_directory_and_deduplicated_cli(self):
        part = self.part(4, 8, [4, 5])
        config = self.path.parent / 'sample.yaml'
        config.write_text(f'output:\n  path: {self.path}\n')
        log = self.path.parent / 'output.log'
        log.write_text(
            f'\x1b[1;32m[info]\x1b[00m Resuming from {self.path}.\n')
        for path in (config, log, part):
            self.assertEqual(status.resolve_inputs(path), [self.path])
        self.assertEqual(status.resolve_inputs(self.path.parent), [self.path])
        log.write_text(f'Sampling ranges for: {self.path}\n')
        self.assertEqual(status.resolve_inputs(log), [self.path])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = status.main([str(config), str(part), str(log), '--ranges'])
        self.assertEqual(code, 0)
        self.assertIn('UNMERGED', output.getvalue())
        self.assertIn('4:8', output.getvalue())
        # Full path in range heading.
        self.assertEqual(output.getvalue().count(str(self.path)), 1)

    def test_inconsistent_legacy_outputs(self):
        with fits.open(self.path, mode='update') as hdus:
            hdus['second'].data = np.zeros((3, 1))
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            status.inspect_dataset(self.path)


if __name__ == '__main__':
    unittest.main()
