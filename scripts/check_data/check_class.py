import classy
import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import time


def get_pk_1(cosmo, k, z, nonlinear=False, only_cb=False):
    """
    Input units: k in h/Mpc, z redshift.
    Output units: P(k) in (Mpc/h)^3
    """
    # Get array of pk
    pk_array, k_array, z_array = cosmo.get_pk_and_k_and_z(
        nonlinear=nonlinear,
        only_clustering_species=only_cb,
        h_units=False)

    # Adjust units
    k_array /= cosmo.h()
    pk_array *= cosmo.h()**3.

    # Flip z_array (for the interpolation it has to be increasing)
    z_array = np.flip(z_array)
    pk_array = np.flip(pk_array, axis=1)

    outside = (k < k_array.min()) | (k > k_array.max())
    print('Native k range [h/Mpc]: [{:.8g}, {:.8g}]; '
          '{} requested points use cubic extrapolation'.format(
              k_array.min(), k_array.max(), np.count_nonzero(outside)))
    if np.min(z) < z_array[0] or np.max(z) > z_array[-1]:
        raise ValueError('Requested redshifts exceed the native CLASS table')

    # Keep cubic k extrapolation explicitly as the diagnostic comparison.
    pk = interp.make_splrep(k_array, pk_array, s=0)(k)

    pk = interp.make_splrep(z_array, pk.T, s=0)(z)

    return pk


def get_pk_2(cosmo, k, z, nonlinear=False, only_cb=False):
    """
    Input units: k in h/Mpc, z redshift.
    Output units: P(k) in (Mpc/h)^3
    """

    if nonlinear is True and only_cb is True:
        fun = cosmo.pk_cb
    elif nonlinear is True and only_cb is False:
        fun = cosmo.pk
    elif nonlinear is False and only_cb is True:
        fun = cosmo.pk_cb_lin
    else:
        fun = cosmo.pk_lin

    pk = np.zeros((len(z), len(k)))

    # Get array of pk
    for nkv, kv in enumerate(k):
        for nzv, zv in enumerate(z):
            pk[nzv, nkv] = fun(kv*cosmo.h(), zv)

    # Adjust units
    pk *= cosmo.h()**3.

    return pk


def get_pk_3(cosmo, k, z, nonlinear=False, only_cb=False):
    """
    Input units: k in h/Mpc, z redshift.
    Output units: P(k) in (Mpc/h)^3
    """

    if nonlinear is True and only_cb is True:
        fun = cosmo.get_pk_cb
    elif nonlinear is True and only_cb is False:
        fun = cosmo.get_pk
    elif nonlinear is False and only_cb is True:
        fun = cosmo.get_pk_cb_lin
    else:
        fun = cosmo.get_pk_lin

    n_mu = 1
    n_z = len(z)
    n_k = len(k)
    k_3D = np.broadcast_to(
        (np.asarray(k) * cosmo.h())[:, None, None],
        (n_k, n_z, n_mu)).copy()
    pk = fun(k_3D, z, n_k, n_z, 1) * cosmo.h()**3.

    return pk[:, :, 0].T


def main():
    idx_data = -1

    # Load reference Pk from file
    fits = io.FitsFile('../emu_like/output/pk_001_thin_no_YHe.fits')

    x_data = fits.get_data('x_data')[idx_data]
    z = np.array([x_data[0]])
    params = {
        'h': x_data[1],
        'Omega_m': x_data[2],
        'Omega_b': x_data[3],
        'tau_reio': x_data[4],
    }

    ref_k = fits.get_data('k_range_pk_m')
    ref_z = fits.get_data('z_array')

    pk_over_pk_ref = fits.get_data('pk_m')[idx_data]
    ref_pk = interp.make_splrep(ref_z, fits.get_data('ref_pk_m')[0].T, s=0)(z)
    pk_data = ref_pk * pk_over_pk_ref

    # Adjust parameters
    args = fits.get_header(0)['y_model']['args']
    args['z_pk'] = max(z)
    args['z_max_pk'] = max(args.get('z_max_pk', 0.1), 0.1, max(z))

    # Init classy
    cosmo = classy.Class()
    cosmo.set(args | params)
    cosmo.compute()

    # Compute Pk_1
    start = time.time()
    pk_1 = get_pk_1(cosmo, ref_k, z, nonlinear=False, only_cb=False)
    print('pk_1 run in {} secs'.format(time.time() - start))
    start = time.time()
    pk_2 = get_pk_2(cosmo, ref_k, z, nonlinear=False, only_cb=False)
    print('pk_2 run in {} secs'.format(time.time() - start))
    start = time.time()
    pk_3 = get_pk_3(cosmo, ref_k, z, nonlinear=False, only_cb=False)
    print('pk_3 run in {} secs'.format(time.time() - start))

    plt.plot(ref_k, np.abs(pk_1[0]/pk_data[0] - 1.)*100.,
             label='Native-table spline (includes extrapolation)')
    plt.plot(ref_k, np.abs(pk_2[0]/pk_data[0] - 1.)*100.,
             label='CLASS scalar evaluator')
    plt.plot(ref_k, np.abs(pk_3[0]/pk_data[0] - 1.)*100.,
             label='CLASS array evaluator')
    _, native_k, _ = cosmo.get_pk_and_k_and_z(nonlinear=False)
    native_min, native_max = native_k.min()/cosmo.h(), native_k.max()/cosmo.h()
    if ref_k.min() < native_min:
        plt.axvspan(ref_k.min(), native_min, color='grey', alpha=.15,
                    label='Spline extrapolation region')
    if ref_k.max() > native_max:
        plt.axvspan(native_max, ref_k.max(), color='grey', alpha=.15)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('perc_rel_diff [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/test_pk_ref.pdf')
    plt.close()


if __name__ == '__main__':
    main()
