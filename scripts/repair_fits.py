import emu_like.io as io

spectra_list = {
    'pk': ['pk_m', 'pk_cb', 'pk_weyl', 'fk_m', 'fk_cb', 'fk_weyl'],
    'cl': ['cl_TT_lensed', 'cl_TE_lensed', 'cl_EE_lensed', 'cl_pp_lensed', 'cl_Tp_lensed', 'cl_BB_lensed'],
}

model = 'lcdm_nu_k'

for spectrum_type in ['cl', 'pk']:
    for parameter_space in ['thin', 'std', 'ext']:

        fits = io.FitsFile('/ceph/hpc/data/s25r06-05-users/{}/sample/{}_100_{}.fits'.format(model, spectrum_type, parameter_space))

        n_samples = []
        for spectrum in spectra_list[spectrum_type]:
            n_samples.append(fits.get_data(spectrum).shape[0])
        n_samples = min(n_samples)

        for spectrum in spectra_list[spectrum_type]:
            data = fits.get_data(spectrum)[:n_samples]
            fits.update(
                name=spectrum,
                data=data,
            )

        fits.print_info()
