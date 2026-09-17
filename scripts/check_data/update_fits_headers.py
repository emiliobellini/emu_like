"""Add dataset metadata without calculating spectra. Dry run unless --apply.

Run from the checkout with PYTHONPATH=src. Inferred metadata follows the
current sampling implementation; it is not proof of historical runtime
settings.
"""
import argparse
from pathlib import Path
import shutil

from astropy.io import fits as afits

from emu_like import io
from emu_like.header_metadata import input_header, spectrum_headers
from emu_like.range_sampling import lock
from emu_like.spectra import Spectra, GrowthRate


def plan(path):
    file = io.FitsFile(str(path))
    primary = file.get_header(0)
    model = primary['y_model']
    if model['name'] != 'class_spectra' or 'params' not in primary:
        raise ValueError('Expected a CLASS sampling dataset')
    spectra = Spectra(model['outputs'])
    refs = [file.get_header('ref_' + sp.name) for sp in spectra]
    if not refs or not refs[0] or any(ref != refs[0] for ref in refs[1:]):
        raise ValueError('Missing or inconsistent reference CLASS headers')
    args = dict(model['args'])
    if any(sp.name in ('pk_weyl', 'fk_weyl') for sp in spectra):
        tokens = args.get('output', '').replace(',', ' ').split()
        args['output'] = ', '.join(dict.fromkeys(tokens + ['mPk', 'wPk']))
    x_header = input_header(primary['params'])
    with afits.open(path, memmap=True) as hdus:
        if hdus['X_DATA'].header['NAXIS1'] != len(x_header['parameters']):
            raise ValueError(
                'X_DATA width disagrees with parameter configuration')
    updates = {}
    old_x = file.get_header('X_DATA')
    if 'parameters' not in old_x:
        updates['X_DATA'] = old_x | x_header
    headers = spectrum_headers(
        spectra, args,
        list(x_header['parameters']),
        GrowthRate.derivative_step,
        provenance='inferred_from_input_and_current_code')
    for sp, header in zip(spectra, headers):
        old = file.get_header(sp.name)
        if 'class_metadata' not in old:
            updates[sp.name.upper()] = old | header
    return updates


def migrate(path, apply=False):
    path = Path(path)
    with lock(path):
        updates = plan(path)
        if not apply or not updates:
            return updates
        backup = path.with_name(path.name + '.before_header_update.bak')
        # Exclusive creation prevents accidentally overwriting a prior backup.
        with backup.open('xb') as target, path.open('rb') as source:
            shutil.copyfileobj(source, target)
        codec = io.FitsFile(str(path))
        with afits.open(path, mode='update', memmap=True) as hdus:
            for hdu in hdus:
                if hdu.name not in updates:
                    # Astropy otherwise refreshes checksum timestamps even
                    # on untouched HDUs when growing headers rewrites a file.
                    # Their bytes and existing checksums remain valid.
                    hdu._output_checksum = False
            for name, header in updates.items():
                encoded = codec._delistify(codec._flatten_dict(header))
                hdu = hdus[name]
                for key in list(hdu.header):
                    if key.startswith('__'):
                        del hdu.header[key]
                hdu.header.update(afits.Header(encoded))
            hdus.flush(output_verify='exception')
    return updates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--apply', action='store_true',
                        help='Update in place, keeping a full .bak copy first')
    options = parser.parse_args()
    for path in options.files:
        changes = migrate(path, apply=options.apply)
        print(path, 'updated' if options.apply else 'would update',
              ', '.join(changes) or '(already current)')
