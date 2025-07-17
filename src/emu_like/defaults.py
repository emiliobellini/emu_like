"""
This is the place that contains all the default
values and names of the code.
"""

# File names
file_names = {
    'params': {
        'name': 'params.yaml',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n'),
    },
    'x_data': {
        'name': 'x_data.txt',
    },
    'y_data': {
        'name': 'y_data_{}.txt',
    },
    'log': {
        'name': 'history_log.cvs',
    },
    'checkpoint': {
        'name': 'checkpoint_epoch{epoch:04d}.weights.h5',
        'folder': 'checkpoints',
    },
    'model': {
        'name': 'model.keras',
    },
    'x_scaler': {
        'name': 'x_scaler.save',
    },
    'y_scaler': {
        'name': 'y_scaler.save',
    },
    'dataset_details': {
        'name': 'dataset_details.yaml',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n'),
    },
    'chains': {
        'name': 'chain.txt',
    },
    'spectra_factor': {
        'name': 'spectra_factor.fits',
    },
}

# Cosmo (Planck 2018 bestfit, Table 1 of https://arxiv.org/pdf/1807.06209)
cosmo_params = {
    'h': 0.6732,
    'Omega_m': 0.3158,
    'Omega_b': 0.0494,
    'A_s': 2.1006e-9,
    'n_s': 0.96605,
    'tau_reio': 0.0543,
    # Precision parameters
    'k_per_decade_for_pk': 1000,
    'k_per_decade_for_bao': 2000,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.01,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'k_max_tau0_over_l_max': 8,
}