"""Dataset metadata, independent of the CLASS runtime."""
import copy


def input_header(params):
    """Ordered column descriptions; priors describe configured, not observed ranges."""
    return {'parameters': {
        name: {'column': index, 'prior': copy.deepcopy(params[name]['prior'])}
        for index, name in enumerate(
            name for name, value in params.items()
            if isinstance(value, dict) and 'prior' in value)}}


def spectrum_headers(spectra, args, names, derivative_step,
                     provenance='recorded_at_generation'):
    fixed = {k: copy.deepcopy(v) for k, v in args.items() if k not in names}
    growth = any(sp.name in ('fk_m', 'fk_cb', 'fk_weyl') for sp in spectra)
    headers = []
    for sp in spectra:
        header = sp.get_header()
        header['class_parameters'] = fixed.copy()
        header['class_metadata'] = {
            'version': 1, 'provenance': provenance,
            'varying_parameters': 'X_DATA.parameters',
        }
        if any(s.is_pk for s in spectra):
            header['class_metadata']['z_max_pk'] = {
                'rule': 'max(minimum, configured, z + padding)',
                'minimum': 0.1, 'configured': args.get('z_max_pk', 0.1),
                'z': 'sample z_pk, or fixed z_pk, or 0',
                'padding': '2*step if z < step else step',
                'step': derivative_step if growth else 0.,
            }
        header['stored_quantity'] = (
            'spectrum / reference' if sp.ratio else 'spectrum')
        if sp.ratio:
            header['stored_units'] = 'dimensionless'
        headers.append(header)
    return headers
