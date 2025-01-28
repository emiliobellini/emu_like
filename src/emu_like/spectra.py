"""
.. module:: spectra

:Synopsis: Spectra computed by Class.
:Author: Emilio Bellini

"""

import numpy as np
import scipy.interpolate as interp
from . import defaults as de


# ----------------- Generic Spectra -----------------------------------------

class Spectra(object):
    """
    This class acts as a container for all the spectra.
    It eases the computation of common quantities.
    """

    def __init__(self, settings, params):
        self.list = [Spectrum.choose_one(sp, settings[sp], params)
                     for sp in settings]
        return

    def __setitem__(self, item, value):
        self.list[item] = value

    def __getitem__(self, item):
        return self.list[item]

    def get_k_min(self):
        """
        Get the global k_min.
        """
        k_min = [x.k_min for x in self.list if x.is_pk]
        try:
            self.k_min = min(k_min)
        except ValueError:
            self.k_min = None
        return self.k_min

    def get_k_max(self):
        """
        Get the global k_max.
        """
        k_max = [x.k_max for x in self.list if x.is_pk]
        try:
            self.k_max = max(k_max)
        except ValueError:
            self.k_max = None
        return self.k_max

    def get_ell_max(self):
        """
        Get the global ell_max.
        """
        ell_max = [x.ell_max for x in self.list if x.is_cl]
        try:
            self.ell_max = max(ell_max)
        except ValueError:
            self.ell_max = None
        return self.ell_max

    def get_want_lensing(self):
        """
        If any of the spectra wants lensing return True.
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
        """
        Get a list of the size of the data
        vector of each spectrum computed.
        """
        n_vecs = [sp.get_n_vec() for sp in self.list]
        return n_vecs

    def get_names(self):
        """
        Get a list of the y names of each spectrum computed.
        """
        names = [sp.get_names() for sp in self.list]
        return names

    def get_headers(self):
        """
        Get a list of the y headers of each spectrum computed.
        """
        headers = [sp.get_header() for sp in self.list]
        return headers

    def get_fnames(self):
        """
        Get a list of the y file names of each spectrum computed.
        """
        fnames = [sp.get_fname() for sp in self.list]
        return fnames


class Spectrum(object):
    """
    Base Spectrum class.
    Return the correct spectrum with the choose_one method,
    and initialise common attributes.
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
        elif spectrum_type == 'pk_weyl':
            return WeylPk(spectrum_type, settings, params)
        elif spectrum_type == 'cl_TT':
            return CellTT(spectrum_type, settings, params)
        else:
            raise ValueError('Spectrum not recognized!')

    def get_fname(self):
        """
        Get the default file name for each spectrum.
        It appends the name of the spectrum
        to the default name.
        """
        fname = de.file_names['y_sample']['name'].format('_' + self.name)
        return fname


# ----------------- Generic Pk and Cell ---------------------------------------

class Pk(Spectrum):
    """
    Generic class for k-dependent power spectra.

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.
    """

    def __init__(self, name, settings, params):
        Spectrum.__init__(self, name, settings, params)
        self.is_pk = True
        self.is_cl = False
        if settings:
            self._init_pk_settings(settings)
        return

    def _init_pk_settings(self, settings):
        """
        Get attributes common to all the k dependent spectra.
        """
        self.k_min = settings['k_min']
        self.k_max = settings['k_max']
        self.k_space = settings['k_space']
        self.k_num = settings['k_num']
        self._get_k_range()
        return

    def _get_k_range(self):
        """
        Return the k range.
        It is possible linear or log spacing.
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
        """
        Return the size of the data vector.
        """
        return self.k_num

    def get_names(self):
        """
        Default names for the P(k_i) at bin i.
        """
        names = ['k_{}'.format(y) for y in range(self.get_n_vec())]
        return names

    def get_header(self):
        """
        Default header for the Pk.
        Format example: {Matter, 1.e-3, 1., log, 600}
        """
        hd = '{} power spectrum P(k) in units (Mpc/h)^3 as a function of '
        hd += 'k (h/Mpc).\nk_min (h/Mpc) = {}, k_max (h/Mpc) = {}, '
        hd += '{}-sampled, for a total number of k_modes of {}.\n\n'
        hd += '\t'.join(self.get_names())
        return hd


class Cl(Spectrum):
    """
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
        """
        Get attributes common to all the ell dependent spectra.
        """
        self.ell_min = settings['ell_min']
        self.ell_max = settings['ell_max']
        self.ell_range = np.arange(self.ell_min, self.ell_max+1)
        self.ell_num = len(self.ell_range)
        return

    def get_n_vec(self):
        """
        Return the size of the data vector.
        """
        return self.ell_num

    def get_names(self):
        """
        Default names for the C(ell_i) at bin i.
        NOTE: bin 0 does not mean ell=0 but ell=2 if ell_min=2.
        """
        names = ['ell_{}'.format(y) for y in range(self.get_n_vec())]
        return names

    def get_header(self):
        """
        Default header for the Cell.
        Format example: {TT, lensed, 2, 2500}
        """
        hd ='dimensionless {} {} [l(l+1)/2pi] C_l for ell={} to {}.\n\n'
        hd += '\t'.join(self.get_names())
        return hd


# ----------------- Pk -----------------------------------------------------

class MatterPk(Pk):
    """
    Matter power spectrum.

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.
    """

    def __init__(self, name, settings, params):
        Pk.__init__(self, name, settings, params)
        # Put here the list of spectra that Class should
        # compute. They should go to the 'output' argument.
        self.class_spectrum = ['mPk']
        return

    def get_header(self):
        """
        Fill the default header.
        """
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
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # Get redshift
        if 'z_pk' in cosmo.pars:
            z_pk = cosmo.pars['z_pk']
        else:
            z_pk = 0.

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # Get pk
        pk = np.array([cosmo.pk(k, z_pk) for k in k_range])

        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.
        return pk


class ColdBaryonPk(Pk):
    """
    CDM+baryon power spectrum.

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.
    """

    def __init__(self, name, settings, params):
        Pk.__init__(self, name, settings, params)
        # Put here the list of spectra that Class should
        # compute. They should go to the 'output' argument.
        self.class_spectrum = ['mPk']
        return

    def get_header(self):
        """
        Fill the default header.
        """
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
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # Get redshift
        if 'z_pk' in cosmo.pars:
            z_pk = cosmo.pars['z_pk']
        else:
            z_pk = 0.

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # Get pk
        pk = np.array([cosmo.pk_cb(k, z_pk) for k in k_range])

        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.
        return pk


class WeylPk(Pk):
    """
    Weyl power spectrum.
    As in Class, we use the convention:
    
    Weyl_pk = matter_pk * ((phi+psi)/2./d_m)**2 * k**4

    The k**4 factor is just a convention. Since there is a factor
    k**2 in the Poisson equation this rescaled Weyl spectrum has
    a shape similar to the matter power spectrum.

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.

    TODO: this is ok at linear order. Beyond that I should check it.
    """

    def __init__(self, name, settings, params):
        Pk.__init__(self, name, settings, params)
        # Put here the list of spectra that Class should
        # compute. They should go to the 'output' argument.
        self.class_spectrum = ['mPk', 'dTk']
        return

    def get_header(self):
        """
        Fill the default header.
        """
        hd = Pk.get_header(self)
        hd = hd.format(
            'Weyl',
            self.k_min,
            self.k_max,
            self.k_space,
            self.k_num
        )
        return hd
    
    def get(self, cosmo):
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # Get redshift
        if 'z_pk' in cosmo.pars:
            z_pk = cosmo.pars['z_pk']
        else:
            z_pk = 0.

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # Get pk
        pk = np.array([cosmo.pk(k, z_pk) for k in k_range])

        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.

        # Get transfer functions to rescale the matter Pk
        tk = cosmo.get_transfer(z=z_pk)
        fac = ((tk['phi'] + tk['psi'])/2./tk['d_m'])**2. * tk['k (h/Mpc)']**4.
        fac = interp.interp1d(tk['k (h/Mpc)'], fac)(self.k_range)

        pk *= fac
        return pk


# ----------------- Cell -----------------------------------------------------

class CellTT(Cl):
    """
    TT power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, settings, params):
        Cl.__init__(self, name, settings, params)
        # Put here the list of spectra that Class should
        # compute. They should go to the 'output' argument.
        self.class_spectrum = ['tCl']
        return

    def get_header(self):
        """
        Fill the default header.
        """
        hd = Cl.get_header(self)
        hd = hd.format(
            'TT',
            '',
            self.ell_min,
            self.ell_max,
        )
        return hd
    
    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """

        # Get cell
        sp = cosmo.raw_cl(lmax=self.ell_max)

        ell = sp['ell'][self.ell_min:]
        cl = sp['tt'][self.ell_min:]

        cl *= ell*(ell+1.)/2./np.pi

        return cl
