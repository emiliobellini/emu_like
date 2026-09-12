"""Compare linear sampling/extraction in emu_like and hi_fast.

Run with the environment containing classy and hiclassy:
    python scripts/check_data/check_cross_package.py

Uses local source checkouts, no emulator weights or existing FITS targets.
Each cosmology is computed once per backend. JSON and NPZ results include
an extra same-backend comparison to separate package and backend differences.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def metrics(left, right, k, z):
    absolute = np.abs(left - right)
    relative = absolute / np.maximum(np.abs(left), 1e-300)
    iz, ik = np.unravel_index(np.argmax(relative), relative.shape)
    return dict(max_absolute=float(absolute.max()),
                max_relative=float(relative.max()),
                worst_k=float(k[ik]), worst_z=float(z[iz]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--hi-fast-src', type=Path,
                        default=root.parent/'hi_fast'/'src')
    parser.add_argument('--output-dir', type=Path,
                        default=root/'output'/'cross_package')
    args = parser.parse_args()
    sys.path[:0] = [str(root/'src'), str(args.hi_fast_src.resolve())]
    import classy
    import hiclassy
    import emu_like.spectra as sampling
    import hi_fast.spectra as extraction
    from hi_fast._class_cache import HiClassCache
    from hi_fast._class_service import HiClassService

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        versions=dict(classy=classy.__version__, hiclassy=hiclassy.__version__),
        paths=dict(classy=classy.__file__, hiclassy=hiclassy.__file__,
                   emu_like=sampling.__file__, hi_fast=extraction.__file__),
        cases=[])
    print(json.dumps(report, indent=2), flush=True)
    z = np.array([0., .0005, .001, .5, 2.])
    names = ('pk_m', 'pk_cb', 'fk_m', 'fk_cb')
    for curvature, omega_k in (('flat', 0.), ('open', .055), ('closed', -.055)):
        for massive in (False, True):
            label = f'{curvature}_' + ('massive' if massive else 'massless')
            params = dict(h=.67, omega_b=.0224, omega_cdm=.12,
                          Omega_k=omega_k, A_s=2.1e-9, n_s=.965,
                          tau_reio=.054, YHe=.24, N_ncdm=int(massive),
                          N_ur=2.0328 if massive else 3.046,
                          output='mPk, dTk', z_max_pk=2.1,
                          k_per_decade_for_pk=40, k_per_decade_for_bao=80,
                          perturbations_sampling_stepsize=.02)
            params['P_k_max_h/Mpc'] = 1.
            if massive:
                params['m_ncdm'] = .06
            native = classy.Class()
            cache = HiClassCache()
            print(f'Computing {label}', flush=True)
            try:
                native.set(params)
                native.compute()
                with cache.use(params) as fast_native:
                    minima = []
                    for backend in (native, fast_native):
                        _, native_k, _ = backend.get_pk_and_k_and_z(
                            nonlinear=False, h_units=False)
                        minima.append(float(native_k.min()/backend.h()))
                    k = np.unique(np.concatenate((
                        np.geomspace(1e-7, 1., 61),
                        *(minimum*np.array([.5, .99, 1., 1.01, 2.])
                          for minimum in minima))))
                    objects = {}
                    for name in names:
                        cls = extraction.Pk if name.startswith('pk') else extraction.Fk
                        obj = cls.__new__(cls)
                        obj.name = name
                        obj.class_args = {}
                        obj.class_high_prec = {}
                        obj._class_cache = cache
                        objects[name] = obj
                    service = HiClassService(objects, cache=cache)
                    coordinates = {'k': k, 'z': z}
                    fast_result = service.get_many(params, {
                        kind: {species: coordinates for species in ('m', 'cb')}
                        for kind in ('pk', 'fk')})
                    case = dict(name=label, params=params,
                                native_k_min=dict(zip(('classy', 'hiclassy'), minima)),
                                comparisons={}, reference_checks={})
                    arrays = dict(k=k, z=z)
                    for name in names:
                        obj = sampling.Spectrum.choose_one(name, dict(
                            k_min=k.min(), k_max=k.max(), k_num=len(k),
                            k_space='log', ratio=False))
                        # Include exact boundary probes on the common grid.
                        obj.k_range = k.copy()
                        if hasattr(obj, 'pk'):
                            obj.pk.k_range = k.copy()
                        emu = np.stack([obj.get(native, float(zi)) for zi in z])
                        emu_same = np.stack([
                            obj.get(fast_native, float(zi)) for zi in z])
                        kind, species = name.split('_')
                        fast = fast_result[kind][species][0]
                        for values in (emu, emu_same, fast):
                            if not np.isfinite(values).all():
                                raise ValueError(f'{label} {name}: nonfinite results')
                        cross = metrics(emu, fast, k, z)
                        same = metrics(emu_same, fast, k, z)
                        case['comparisons'][name] = dict(
                            cross_backend=cross, same_backend=same)
                        np.testing.assert_allclose(emu_same, fast,
                                                   rtol=1e-9, atol=1e-11)
                        # Reference tables must agree with scalar evaluation
                        # on their own grid, including both endpoints.
                        reference = obj.get(native)
                        selected = [0, len(obj.z_array)//2, len(obj.z_array)-1]
                        reference_error = 0.
                        for j in selected:
                            single = obj.get(native, float(obj.z_array[j]))
                            reference_error = max(reference_error, float(
                                np.max(abs(reference[:, j]-single))))
                            np.testing.assert_allclose(reference[:, j], single,
                                                       rtol=1e-10, atol=1e-11)
                        case['reference_checks'][name] = reference_error
                        arrays[name+'_emu_like'] = emu
                        arrays[name+'_hi_fast'] = fast
                        arrays[name+'_emu_like_hiclassy'] = emu_same
                        print(f'  {name}: cross-backend max relative '
                              f'{cross["max_relative"]:.3e}; same-backend '
                              f'{same["max_relative"]:.3e}', flush=True)
                    assert cache.info()['misses'] == 1, cache.info()
                    case['hiclassy_computations'] = cache.info()['misses']
                    np.savez_compressed(args.output_dir/(label+'.npz'), **arrays)
                    report['cases'].append(case)
                    (args.output_dir/'report.json').write_text(
                        json.dumps(report, indent=2)+'\n')
            finally:
                native.struct_cleanup()
                native.empty()
                cache.clear()
    print(f'Report: {args.output_dir / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
