"""Plot destinations preserve distinct standard and Sobolev training runs."""

from pathlib import Path
import tempfile
import unittest

from scripts.check_train._plot_paths import plot_path, source_root


class PlotPathTests(unittest.TestCase):
    def test_flatten_nested_runs_for_every_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / 'train'
            for relative in ('pk_m', 'sobolev/pk_m'):
                run = train / relative
                run.mkdir(parents=True)
                (run / 'history_log.csv').touch()
                for filename in ('loss_vs_epoch.png', 'histograms.png',
                                 'summary_table.txt', 'worst_modes.png'):
                    name = Path(filename)
                    label = relative.replace('/', '_')
                    expected = root / 'plots' / (
                        f'{name.stem}_{label}{name.suffix}')
                    self.assertEqual(
                        plot_path(run, root / 'plots', filename,
                                  source_root([train])), expected)
                    self.assertTrue(expected.parent.is_dir())
                    self.assertEqual(
                        plot_path(run, root / 'plots', filename), expected)
                    self.assertEqual(
                        plot_path(run, None, filename), run / filename)

    def test_direct_and_multiple_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / 'train'
            standard = train / 'pk_m'
            sobolev = train / 'sobolev' / 'pk_m'
            for run in (standard, sobolev):
                run.mkdir(parents=True)
                (run / 'history_log.csv').touch()
            self.assertEqual(source_root([str(standard) + '/']), train)
            self.assertEqual(source_root([standard, sobolev]), train)


if __name__ == '__main__':
    unittest.main()
