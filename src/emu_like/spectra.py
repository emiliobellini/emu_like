"""
.. module:: spectra

:Synopsis: Spectra computed by Class.
:Author: Emilio Bellini

"""

import numpy as np
from . import defaults as de


class Spectra(object):
    """
    This class acts as a container for all the spectra.
    It eases the computation of common quantities.
    """

    def __init__(self, settings, params):
        self.list = [Spectrum.choose_one(sp, settings[sp], params)
                     for sp in settings]
        pass

    def __setitem__(self, item, value):
        self.list[item] = value

    def __getitem__(self, item):
        return self.list[item]

    def get_k_min(self):
        """TODO
        """
        k_min = [x.k_min for x in self.list if x.is_pk]
        try:
            self.k_min = min(k_min)
        except ValueError:
            self.k_min = None
        return self.k_min

    def get_k_max(self):
        """TODO
        """
        k_max = [x.k_max for x in self.list if x.is_pk]
        try:
            self.k_max = max(k_max)
        except ValueError:
            self.k_max = None
        return self.k_max

    def get_ell_max(self):
        """TODO
        """
        ell_max = [x.ell_max for x in self.list if x.is_cl]
        try:
            self.ell_max = max(ell_max)
        except ValueError:
            self.ell_max = None
        return self.ell_max

    def get_want_lensing(self):
        """TODO
        """
        want_lensing = [x.want_lensing for x in self.list if x.is_cl]
        try:
            self.want_lensing = any(want_lensing)
        except ValueError:
            self.want_lensing = False
        return self.want_lensing

    def get_class_output(self):
        """
        Add keys to the parameters passed to class
        to have a consistent evolution with all the
        spectra/settings requested.
        """
        class_dict = {}

        # Output spectra
        class_output = [x.class_spectrum for x in self]
        class_output = [x for xs in class_output for x in xs]
        class_output = list(set(class_output))
        if class_output:
            class_dict['output'] = ', '.join(class_output)

        # k max Pk
        if self.get_k_max():
            class_dict['P_k_max_h/Mpc'] = self.k_max

        # ell max Pk
        if self.get_ell_max():
            class_dict['l_max_scalars'] = self.ell_max

        # lensing
        if self.get_want_lensing():
            class_dict['lensing'] = self.want_lensing

        return class_dict

    def get_n_vecs(self):
        n_vecs = [sp.get_n_vec() for sp in self.list]
        return n_vecs

    def get_names(self):
        names = [sp.get_names() for sp in self.list]
        return names

    def get_headers(self):
        headers = [sp.get_header() for sp in self.list]
        return headers

    def get_fnames(self):
        fnames = [sp.get_fname() for sp in self.list]
        return fnames


class Spectrum(object):
    """
    Base Spectrum class.
    TODO
    It is use to rescale data (useful for training to
    have order 1 ranges). This main class has three
    main methods:
    - choose_one: redirects to the correct subclass
    - load: load a scaler from a file
    - save: save scaler to a file.

    Each one of the other scalers (see below), should
    inherit from this and define three other methods:
    - fit: fit scaler
    - transform: transform data using fitted scaler
    - inverse_transform: transform back data.
    """

    def __init__(self, name, settings, params):
        self.name = name
        self.settings = settings
        self.params = params
        return

    @staticmethod
    def choose_one(spectrum_type, settings, params):
        """
        Main function to get the correct Spectrum.

        Arguments:
            - spectrum_type (str): type of spectrum.

        Return:
            - Spectrum (object): get the correct
              spectrum and initialize it.
        """
        if spectrum_type == 'pk_m':
            return MatterPk(spectrum_type, settings, params)
        elif spectrum_type == 'pk_cb':
            return ColdBaryonPk(spectrum_type, settings, params)
        else:
            raise ValueError('Spectrum not recognized!')

    def get_fname(self):
        fname = de.file_names['y_sample']['name'].format('_' + self.name)
        return fname


class Pk(Spectrum):
    """
    TODO
    Generic class for k-dependent power spectra.

    NOTE: all the units are in powers of h/Mpc.
    """

    def __init__(self, name, settings, params):
        Spectrum.__init__(self, name, settings, params)
        self.is_pk = True
        self.is_cl = False
        if settings:
            self._init_pk_settings(settings, params)
        return

    def _init_pk_settings(self, settings, params):
        """TODO
        """
        self.k_min = settings['k_min']
        self.k_max = settings['k_max']
        self.k_space = settings['k_space']
        self.k_num = settings['k_num']
        self._get_k_range()
        return

    def _get_k_range(self):
        """TODO
        """
        if self.k_space == 'linear':
            fun = np.linspace
            start = self.k_min
            stop = self.k_max
        elif self.k_space == 'log':
            fun = np.logspace
            start = np.log10(self.k_min)
            stop = np.log10(self.k_max)
        else:
            raise Exception('Spacing not recognized!')
        self.k_range = fun(start, stop, num=self.k_num)
        return self.k_range
    
    def get_n_vec(self):
        return self.k_num

    def get_names(self):
        names = ['k_{}'.format(y) for y in range(self.get_n_vec())]
        return names

    def get_header(self):
        hd = '{} power spectrum P(k) in units (Mpc/h)^3 as a function of '
        hd += 'k (h/Mpc).\nk_min (h/Mpc) = {}, k_max (h/Mpc) = {}, '
        hd += '{}-sampled, for a total number of k_modes of {}.\n\n'
        hd += '\t'.join(self.get_names())
        return hd

class Cl(Spectrum):
    """
    TODO
    Generic class for ell-dependent power spectra.
    """

    def __init__(self, name, settings, params):
        Spectrum.__init__(self, name, settings, params)
        self.is_pk = False
        self.is_cl = True
        self.want_lensing = False
        if settings:
            self._init_cl_settings(settings)
        return

    def _init_cl_settings(self, settings):
        """TODO
        """
        self.ell_min = settings['ell_min']
        self.ell_max = settings['ell_max']
        self.ell_num = self.ell_max - self.ell_max + 1
        return

    def get_n_vec(self):
        return self.ell_num

    def get_names(self):
        names = ['ell_{}'.format(y) for y in range(self.get_n_vec())]
        return names

    def get_header(self):
        # format example (TT, lensed, 0, 2500)
        hd ='dimensionless {} {} [l(l+1)/2pi] C_l for ell={} to {}.\n\n'
        hd += '\t'.join(self.get_names())
        return hd


class MatterPk(Pk):
    """
    TODO
    Matter power spectrum.

    NOTE: all the units are in powers of h/Mpc.
    """

    def __init__(self, name, settings, params):
        Pk.__init__(self, name, settings, params)
        self.class_spectrum = ['mPk']
        return

    def get_header(self):
        hd = Pk.get_header(self)
        hd = hd.format(
            'Total matter',
            self.k_min,
            self.k_max,
            self.k_space,
            self.k_num
        )
        return hd
    
    def get(self, cosmo):

        # Get redshift
        if 'z_pk' in cosmo.pars:
            z_pk = cosmo.pars['z_pk']
        else:
            z_pk = 0.

        # convert k in units of 1/Mpc
        self.k_range *= cosmo.h()

        # Get pk
        pk = np.array([cosmo.pk(k, z_pk) for k in self.k_range])

        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.
        return pk

class ColdBaryonPk(Pk):
    """
    TODO
    CDM+baryon power spectrum.

    NOTE: all the units are in powers of h/Mpc.
    """

    def __init__(self, name, settings, params):
        Pk.__init__(self, name, settings, params)
        self.class_spectrum = ['mPk']
        return

    def get_header(self):
        hd = Pk.get_header(self)
        hd = hd.format(
            'CDM + baryons',
            self.k_min,
            self.k_max,
            self.k_space,
            self.k_num
        )
        return hd

    def get(self, cosmo):

        # Get redshift
        if 'z_pk' in cosmo.pars:
            z_pk = cosmo.pars['z_pk']
        else:
            z_pk = 0.

        # convert k in units of 1/Mpc
        self.k_range *= cosmo.h()

        # Get pk
        pk = np.array([cosmo.pk_cb(k, z_pk)
                       for k in self.k_range])
        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.
        return pk
