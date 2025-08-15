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
# 'lcdm_pk': [2.6041182  0.58768946 0.27363048 0.05497547 0.09483284]
# 'lcdm_nu_cl': [0.70837614 0.31269437 0.0469087  3.07445564 0.95587349 0.05014624 0.09975079 0.02785967]
params = {
    'z_max_pk': 2.604118,
    'h': 0.58768946,
    'Omega_m': 0.27363048,
    'Omega_b': 0.05497547,
    'tau_reio': 0.09483284,
    'm_ncdm': 0.02,
    'ln_A_s_1e10': 3.044,
    'n_s': 0.966,

    'YHe': 0.24,
    'N_ur': 0.,
    'N_ncdm': 1,
    'deg_ncdm': 3,
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
    'k_pivot': 0.05,
    'modes': 's',
}

cosmo.set(params)
cosmo.compute()
cl_1 = cosmo.raw_cl()
cl_1l = cosmo.lensed_cl()
k = np.logspace(-4., 1., num=600)
pk_1 = get_pk(cosmo, k)
exit()

cosmo = classy.Class()
params = {

    'z_max_pk': 2.604118,
    'h': 0.58768946,
    'Omega_m': 0.27363048,
    'Omega_b': 0.05497547,
    'tau_reio': 0.09483284,
    'm_ncdm': 0.02,
    'ln_A_s_1e10': 3.044,
    'n_s': 0.966,

    'YHe': 0.24,
    'N_ur': 0.,
    'N_ncdm': 1,
    'deg_ncdm': 3,
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
    'k_pivot': 0.05,
    'modes': 's',
}
cosmo.set(params)
cosmo.compute()
cl_2 = cosmo.raw_cl()
cl_2l = cosmo.lensed_cl()
pk_2 = get_pk(cosmo, k)


# plt.plot(cl_1['ell'], cl_1['ell']*(cl_1['ell']+1)/2/np.pi*cl_1['tt'])
# plt.plot(cl_1['ell'], cl_1['ell']*(cl_1['ell']+1)/2/np.pi*cl_2['tt'], '--')
# plt.plot(cl_1['ell'], cl_1['tt']/cl_2['tt'])
# plt.plot(cl_1l['ell'], cl_1l['tt']/cl_2l['tt'], '--')

# plt.plot(k, pk_1)
# plt.plot(k, pk_2)
plt.plot(k, pk_2/pk_1)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('scripts/test.pdf')

print(100*np.max(np.abs(pk_1/pk_2-1.)))
print(100*np.min(np.abs(pk_1/pk_2-1.)))


# print(100*np.max(np.abs(cl_1['tt']/cl_2['tt']-1.)))
# print(100*np.min(np.abs(cl_1['tt']/cl_2['tt']-1.)))
