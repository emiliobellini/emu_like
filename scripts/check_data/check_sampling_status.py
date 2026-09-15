"""Compact saved-row counts from main FITS files and range checkpoints.

Accept main FITS, sampling YAML, sampling stdout logs, or directories of FITS.
No model initialization or spectrum-array loading is required.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np
from astropy.io import fits
import yaml
import emu_like.io as io

RANGE_FILE = re.compile(r'^(.*)\.rows-\d+-\d+\.fits$')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def main_path(path):
    """Map a numbered checkpoint to the conventional main FITS filename."""
    path = Path(path)
    match = RANGE_FILE.match(path.name)
    return path.with_name(match[1] + '.fits') if match else path


def resolve_inputs(path):
    path = Path(path)
    if path.is_dir():
        return sorted({main_path(p) for p in path.glob('*.fits')})
    if path.suffix.lower() in ('.yaml', '.yml'):
        with path.open() as stream:
            return [Path(yaml.safe_load(stream)['output']['path'])]
    if path.suffix.lower() == '.fits':
        return [main_path(path)]
    # Keep accepting sampling stdout logs without requiring tqdm stderr logs.
    # Read only the beginning and end, even for very large logs.
    with path.open('rb') as stream:
        head = stream.read(65536)
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        tail = stream.read()
    text = ANSI.sub('', (head + b'\n' + tail).decode(errors='replace'))
    matches = re.findall(
        r'(?:Resuming from |Writing output in |Range checkpoint: '
        r'|Sampling ranges for: )(.+?\.fits)(?:\.|\s|$)',
        text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(
            'No FITS path found; pass the main FITS or sampling YAML instead')
    return list(dict.fromkeys(main_path(p.strip()) for p in matches))


def inspect_dataset(path):
    """Count unique saved row IDs.

    Read only headers, mask and row-ID arrays.
    """
    path = Path(path)
    helper = io.FitsFile(str(path))
    with fits.open(path, memmap=False) as hdus:
        settings = helper._unflatten_dict(helper._listify(hdus[0].header))
        outputs = settings['y_model']['outputs']
        keys = list(outputs) if isinstance(outputs, dict) else ['y_data']
        total = hdus['x_data'].header['NAXIS2']
        if not keys:
            raise ValueError('No configured outputs')
        widths = {key: hdus[key].header['NAXIS1']
                  for key in keys if key in hdus}
        lengths = [hdus[key].header['NAXIS2'] if key in hdus else 0
                   for key in keys]
        if len(set(lengths)) != 1 or lengths[0] > total:
            raise ValueError(
                'Inconsistent output lengths; a legacy writer may be saving, '
                'retry later')
        if 'SAMPLE_DONE' in hdus:
            data = hdus['SAMPLE_DONE'].data
            if (data.shape != (total,)
                    or not np.isin(data, [0, 1]).all()
                    or lengths[0] != total):
                raise ValueError('Invalid completion mask or output lengths')
            saved = data.astype(bool)
        else:
            saved = np.arange(total) < lengths[0]
    main_count = int(saved.sum())
    ranges, errors = [], []
    # No source hash recalculation: this is a lightweight progress report.
    # The merge command performs full dataset identity/checksum validation.
    for part in sorted(path.parent.glob(path.stem + '.rows-*.fits')):
        try:
            with fits.open(part, memmap=False) as hdus:
                start, stop = hdus[0].header['START'], hdus[0].header['STOP']
                rows = hdus['ROW_IDS'].data
                if (not (0 <= start < stop <= total)
                        or rows.ndim != 1 or rows.dtype.kind not in 'iu'):
                    raise ValueError('Invalid range/row IDs')
                if (np.any((rows < start) | (rows >= stop))
                        or len(np.unique(rows)) != len(rows)):
                    raise ValueError('Out-of-range or duplicate row IDs')
                for key in keys:
                    header = hdus[key].header
                    if header['NAXIS'] != 2 or header['NAXIS2'] != len(rows):
                        raise ValueError(
                            'Output length does not match row IDs')
                    if key in widths and header['NAXIS1'] != widths[key]:
                        raise ValueError(
                            'Output width does not match main FITS')
                saved[rows] = True
                ranges.append((start, stop, len(rows)))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f'{part}: {error}')
    unique = int(saved.sum())
    return dict(path=str(path), total=total, main=main_count,
                unmerged=unique-main_count, saved=unique, missing=total-unique,
                ranges=ranges, errors=errors)


def print_table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in rows))
              for i, header in enumerate(headers)]
    print('  '.join(header.ljust(width)
                    for header, width in zip(headers, widths)))
    for row in rows:
        print('  '.join(value.ljust(width) if i == 0 else value.rjust(width)
                        for i, (value, width) in enumerate(zip(row, widths))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'inputs', nargs='+',
        help='Main/range FITS, sampling YAML, stdout log, or FITS directory')
    parser.add_argument(
        '--ranges', action='store_true',
        help='Also show saved progress for each range checkpoint')
    args = parser.parse_args(argv)
    paths, errors, reports = {}, [], []
    for entry in args.inputs:
        try:
            resolved = resolve_inputs(entry)
            if not resolved:
                raise ValueError('No FITS files found')
            for path in resolved:
                paths.setdefault(path.resolve(), path)
        except (OSError, ValueError, KeyError, TypeError,
                yaml.YAMLError) as error:
            errors.append(f'{entry}: {error}')
    for path in paths.values():
        try:
            report = inspect_dataset(path)
            reports.append(report)
            errors.extend(report['errors'])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f'{path}: {error}')
    if reports:
        names = [Path(r['path']).name for r in reports]
        labels = [name if names.count(name) == 1 else r['path']
                  for name, r in zip(names, reports)]
        print_table(
            ['DATASET', 'TOTAL', 'MAIN', 'UNMERGED', 'SAVED', '%',
             'MISSING', 'FILES'], [
                [label, f"{r['total']:,}", f"{r['main']:,}",
                 f"{r['unmerged']:,}", f"{r['saved']:,}",
                 f"{100*r['saved']/r['total']:.1f}" if r['total'] else '100.0',
                 f"{r['missing']:,}", len(r['ranges'])]
                for label, r in zip(labels, reports)])
        print('\nSAVED = unique rows in main + checkpoints; '
              'UNMERGED = additional rows outside main.')
        print('Counts update at checkpoints. Unsaved work and ranges '
              'without a checkpoint are not shown.')
        if args.ranges:
            for report in reports:
                print(f"\n{report['path']} — range checkpoints "
                      '(stop exclusive)')
                if not report['ranges']:
                    print('No range checkpoints.')
                    continue
                print_table(['RANGE', 'SAVED', 'SIZE', '%', 'UNSAVED'], [
                    [f'{start}:{stop}', f'{count:,}', f'{stop-start:,}',
                     f'{100*count/(stop-start):.1f}', f'{stop-start-count:,}']
                    for start, stop, count in report['ranges']])
    for error in errors:
        print(f'Warning: {error}', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
