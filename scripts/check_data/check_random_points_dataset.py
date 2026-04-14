import argparse
import classy
import numpy as np
import scipy.interpolate as interp
import tqdm
from emu_like.datasets import DataCollection


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('--number-points', '-n', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # Init local variables
    is_diff = False
    count_diff = 0

    # Load data
    data = DataCollection()
    data.load(args.input_file)

    # Generate random points
    random_indices = np.random.choice(
        data.n_samples,
        size=args.number_points,
        replace=False)
    # random_indices = [0, 85713]  # TODO

    cosmo = classy.Class()
    class_params = data.y_model.class_params
    spectra = data.y_model.spectra

    # Get and compare reference spectra
    # 1) Infer the maximum redshift
    if any([sp.is_pk for sp in spectra]):
        z_max = {'z_max_pk': data.y_model._get_z_max()}
    else:
        z_max = {}
    # 2) Compute Class
    cosmo_ref = classy.Class()
    ref_params = data.y_model.ref_params
    ref_params = ref_params | z_max
    cosmo_ref.set(ref_params)
    cosmo_ref.compute()
    # 3) Compute all the spectra
    y_ref = [sp.get(cosmo_ref, z=None)[np.newaxis] for sp in spectra]
    # 4) Replace with ones if we do not take ratio
    for nsp, sp in enumerate(spectra):
        if not sp.ratio:
            y_ref[nsp] = np.ones_like(y_ref[nsp])
    # 5) Store the redshift values at which all Pk have been computed
    z_array = data.y_model._get_z_array(spectra)
    # 6) Store the k modes values at which all Pk have been computed
    k_ranges = [None for sp in spectra]
    for nsp, sp in enumerate(spectra):
        try:
            k_ranges[nsp] = sp.k_range
        except AttributeError:
            pass
    # 7) Store the ell modes values at which all Cell have been computed
    ell_ranges = [None for sp in spectra]
    for nsp, sp in enumerate(spectra):
        try:
            ell_ranges[nsp] = sp.ell_range
        except AttributeError:
            pass

    y_ref_2 = data.y_model.y_ref

    for nsp, sp in enumerate(spectra):
        print('Max relative difference for reference spectrum {}: {:.2e}'
              ''.format(
                  data.y_model.spectra.names[nsp],
                  np.max(np.abs(y_ref[nsp]/y_ref_2[nsp] - 1.))))
    print()

    # Iterate over random indices
    for idx in tqdm.tqdm(random_indices):
        count_diff_per_spectrum = 0
        for key, val in zip(data.x_names, data.x[idx]):
            class_params[key] = val
        cosmo.set(class_params)
        cosmo.compute()

        for nsp, spectrum in enumerate(spectra):
            # Interpolate over z or not
            if spectrum.is_cl:
                y_ref = data.y_model.y_ref[nsp][0]
                recalc_data = spectrum.get(cosmo)/y_ref
            else:
                y_ref = data.y_model.y_ref[nsp][0]
                y_ref = interp.make_splrep(
                    data.y_model.z_array, y_ref.T, s=0)(class_params['z_pk']).T
                recalc_data = spectrum.get(cosmo, z=class_params['z_pk'])/y_ref
            abs_diff = np.abs(recalc_data - data.y[nsp][idx])

            if np.any(abs_diff > 1e-4):
                is_diff = True
                count_diff_per_spectrum += 1
                if args.verbose:
                    print(
                        'Discrepancy found for sample index {} and '
                        'spectrum {} (max abs diff: {:.2e})'.format(
                            idx,
                            data.y_model.spectra.names[nsp],
                            np.max(abs_diff)))
        if count_diff_per_spectrum > 0:
            count_diff += 1
    # Print summary
    if is_diff is False:
        print('No discrepancies found in {} random points.'.format(
            args.number_points))
    else:
        print('Discrepancies found in {} out of {} random points.'.format(
            count_diff,
            args.number_points))
