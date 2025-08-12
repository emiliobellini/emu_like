import classy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

def get_pk(cosmo, k, z=0):
    # Get array of pk
    pk_array, k_array, z_array = cosmo.get_pk_and_k_and_z(
        nonlinear=False,
        only_clustering_species = False,
        h_units=False)

    # Flip z_array (for the interpolation it has to be increasing)
    z_array = np.flip(z_array)
    pk_array = np.flip(pk_array, axis=1)

    # Evaluate pk at the requested range
    pk = interp.make_splrep(k_array, pk_array, s=0)(k)

    pk = interp.make_splrep(z_array, pk.T, s=0)(z)
    return pk

cosmo = classy.Class()
params = {
    'A_s':2.1e-9,
    'n_s':0.966,
    'YHe': 0.24,
    'k_per_decade_for_pk': 40,
    'k_per_decade_for_bao': 80,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.02,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'output': 'tCl, dTk, pCl, lCl, mPk',
    'l_max_scalars': 3000,
    'lensing': 'yes',
    'P_k_max_h/Mpc': 50.0,
    'modes': 's',
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'N_ur': 2.0308,
    'N_ncdm': 1,
    'm_ncdm': 0.06
}
cosmo.set(params)
cosmo.compute()
cl_1 = cosmo.raw_cl()
cl_1l = cosmo.lensed_cl()
k = np.logspace(-4., 1., num=600)
pk_1 = get_pk(cosmo, k)


cosmo = classy.Class()
params = {
    'A_s':2.1e-9,
    'n_s':0.966,
    'YHe': 0.24,
    'k_per_decade_for_pk': 40,
    'k_per_decade_for_bao': 80,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.02,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'output': 'tCl, dTk, pCl, lCl, mPk',
    'l_max_scalars': 3000,
    'lensing': 'yes',
    'P_k_max_h/Mpc': 50.0,
    'modes': 's',
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'N_ur': 0.,
    'N_ncdm': 1,
    'deg_ncdm': 3,
    'm_ncdm': 0.02
}
cosmo.set(params)
cosmo.compute()
cl_2 = cosmo.raw_cl()
cl_2l = cosmo.lensed_cl()
pk_2 = get_pk(cosmo, k)


cosmo = classy.Class()
params = {
    'A_s':2.1e-9,
    'n_s':0.966,
    'YHe': 0.24,
    'k_per_decade_for_pk': 40,
    'k_per_decade_for_bao': 80,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.02,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'output': 'tCl, dTk, pCl, lCl, mPk',
    'l_max_scalars': 3000,
    'lensing': 'yes',
    'P_k_max_h/Mpc': 50.0,
    'modes': 's',
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'N_ur': 0.,
    'N_ncdm': 1,
    'deg_ncdm': 3,
    'm_ncdm': 0.06
}
cosmo.set(params)
cosmo.compute()
cl_3 = cosmo.raw_cl()
cl_3l = cosmo.lensed_cl()
pk_3 = get_pk(cosmo, k)


cosmo = classy.Class()
params = {
    'A_s':2.1e-9,
    'n_s':0.966,
    'YHe': 0.24,
    'k_per_decade_for_pk': 40,
    'k_per_decade_for_bao': 80,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.02,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'output': 'tCl, dTk, pCl, lCl, mPk',
    'l_max_scalars': 3000,
    'lensing': 'yes',
    'P_k_max_h/Mpc': 50.0,
    'modes': 's',
    'tau_reio': 0.04,
    'z_max_pk': 0.1,
    'N_ur': 2.0308,
    'N_ncdm': 1,
    'm_ncdm': 0.18
}
cosmo.set(params)
cosmo.compute()
cl_4 = cosmo.raw_cl()
cl_4l = cosmo.lensed_cl()
k = np.logspace(-4., 1., num=600)
pk_4 = get_pk(cosmo, k)

# plt.plot(cl_1['ell'], cl_1['ell']*(cl_1['ell']+1)/2/np.pi*cl_1['tt'])
# plt.plot(cl_1['ell'], cl_1['ell']*(cl_1['ell']+1)/2/np.pi*cl_2['tt'], '--')
# plt.plot(cl_1['ell'], cl_1['tt']/cl_2['tt'])
# plt.plot(cl_1l['ell'], cl_1l['tt']/cl_2l['tt'], '--')

# plt.plot(k, pk_1)
# plt.plot(k, pk_2)
plt.plot(k, pk_2/pk_1)
plt.plot(k, pk_3/pk_1)
plt.plot(k, pk_4/pk_1)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('scripts/test.pdf')

print(100*np.max(np.abs(pk_1/pk_2-1.)))
print(100*np.min(np.abs(pk_1/pk_2-1.)))


# print(100*np.max(np.abs(cl_1['tt']/cl_2['tt']-1.)))
# print(100*np.min(np.abs(cl_1['tt']/cl_2['tt']-1.)))
