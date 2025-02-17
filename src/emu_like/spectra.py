"""
.. module:: spectra

:Synopsis: Spectra computed by Class.
:Author: Emilio Bellini

: Description: this module provides a list of classes for each
spectrum that can be computed. It uses classy to get the spectra,
the syntax and conventions are equivalent to those in Class. 
Here we are assuming that a classy.Class() object has already
been initialised, and the output has been computed. This is done
in src/emu_like/y_models.py.
"""

import classy
import numpy as np
import scipy.interpolate as interp
from . import defaults as de


# ----------------- Generic Spectra ------------------------------------------#

class Spectra(object):
    """
    This class acts as a container for all the spectra.
    It is useful to get common properties of the spectra,
    as well as to prepare the parameters that should be
    passed to classy for a proper run.
    """

    def __init__(self, params):
        """
        Init the list of spectra to be computed.
        Arguments:
        - params (dict): nested dictionary of parameters
          for each spectrum.
        """
        self.list = [Spectrum.choose_one(sp, params[sp]) for sp in params]
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

    def get_class_params(self):
        """
        Build a dictionary of parameters needed by Class
        to properly compute the spectra requested.
        """
        class_params = {}

        # Output spectra
        class_output = [x.class_spectra for x in self]
        class_output = [x for xs in class_output for x in xs]
        class_output = list(set(class_output))
        if class_output:
            class_params['output'] = ', '.join(class_output)

        # k max Pk
        if self.get_k_max():
            class_params['P_k_max_h/Mpc'] = self.k_max

        # ell max Pk
        if self.get_ell_max():
            class_params['l_max_scalars'] = self.ell_max

        # lensing
        if self.get_want_lensing():
            class_params['lensing'] = 'yes'
            class_params['modes'] = 's'

        return class_params

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

    def __init__(self, name, params):
        # Name of the spectrum as it is in the
        # parameter file (used for the y file names)
        self.name = name
        # Ratio
        self.ratio = False
        # Spectrum dependent parameters, i.e. k_min, k_max, ell_max, ...
        self.params = params
        # Get ratio
        try:
            self.ratio = self.params['ratio']
        except KeyError:
            self.ratio = False
        return

    @staticmethod
    def choose_one(spectrum_type, params):
        """
        Main function to get the correct Spectrum.

        Arguments:
            - spectrum_type (str): type of spectrum.

        Return:
            - Spectrum (object): get the correct
              spectrum and initialize it.
        """
        # Pk
        if spectrum_type == 'pk_m':
            return MatterPk(spectrum_type, params)
        elif spectrum_type == 'pk_cb':
            return ColdBaryonPk(spectrum_type, params)
        elif spectrum_type == 'pk_weyl':
            return WeylPk(spectrum_type, params)
        # Cl
        elif spectrum_type == 'cl_TT':
            return CellTT(spectrum_type, params)
        elif spectrum_type == 'cl_EE':
            return CellEE(spectrum_type, params)
        elif spectrum_type == 'cl_TE':
            return CellTE(spectrum_type, params)
        elif spectrum_type == 'cl_BB':
            return CellBB(spectrum_type, params)
        elif spectrum_type == 'cl_pp':
            return Cellpp(spectrum_type, params)
        elif spectrum_type == 'cl_Tp':
            return CellTp(spectrum_type, params)
        # Cl - lensed
        elif spectrum_type == 'cl_TT_lensed':
            return CellTTLensed(spectrum_type, params)
        elif spectrum_type == 'cl_EE_lensed':
            return CellEELensed(spectrum_type, params)
        elif spectrum_type == 'cl_TE_lensed':
            return CellTELensed(spectrum_type, params)
        elif spectrum_type == 'cl_BB_lensed':
            return CellBBLensed(spectrum_type, params)
        else:
            raise ValueError(
                'Spectrum {} not recognized!'.format(spectrum_type))

    def get_fname(self):
        """
        Get the default file name for each spectrum.
        It appends the name of the spectrum to the default name.
        """
        fname = de.file_names['y_data']['name'].format('_' + self.name)
        return fname

    def _get_range(self, min, max, num, space):
        """
        Return a range given specifics.
        Arguments:
        - min (float): minimum value;
        - max (float): maximum value;
        - num (int): number of elements;
        - space (str): spacing (linear or log).
        """
        if space == 'linear':
            fun = np.linspace
            start = min
            stop = max
        elif space == 'log':
            fun = np.logspace
            start = np.log10(min)
            stop = np.log10(max)
        else:
            raise Exception('Spacing not recognized!')
        rg = fun(start, stop, num=num)
        return rg


# ----------------- Generic Pk and Cell --------------------------------------#

class Pk(Spectrum):
    """
    Generic class for k-dependent power spectra (matter, cb, weyl).

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.
    """

    def __init__(self, name, params):
        Spectrum.__init__(self, name, params)

        # Bools to pick the spectrum type
        self.is_pk = True
        self.is_cl = False

        # Tipical parameters of k dependent PS
        self.k_min = params['k_min']
        self.k_max = params['k_max']
        self.k_num = params['k_num']
        self.k_space = params['k_space']
        self.k_range = self._get_range(
            self.k_min, self.k_max, self.k_num, self.k_space)

        # - (str) name you want to appear in the header of the
        #   file, see Pk.get_header
        self.hd_name = None
        return

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
        if self.ratio:
            hd = 'Ratio of the {} power spectrum P(k) w.r.t. the reference P(k) '
        else:
            hd = '{} power spectrum P(k) in units (Mpc/h)^3 '
        hd += 'as a function of k (h/Mpc).\nk_min (h/Mpc) = {}, '
        hd += 'k_max (h/Mpc) = {}, {}-sampled, for a total number '
        hd += 'of k_modes of {}.\n'
        # hd += '\t'.join(self.get_names())

        hd = hd.format(
            self.hd_name,
            self.k_min,
            self.k_max,
            self.k_space,
            self.k_num
        )
        return hd


class Cell(Spectrum):
    """
    Generic class for ell-dependent power spectra.
    """

    def __init__(self, name, params):
        Spectrum.__init__(self, name, params)

        # Bools to pick the spectrum type
        self.is_pk = False
        self.is_cl = True

        # Tipical parameters of ell dependent PS
        self.ell_min = params['ell_min']
        self.ell_max = params['ell_max']
        self.ell_range = np.arange(self.ell_min, self.ell_max+1)
        self.ell_num = len(self.ell_range)

        # Placeholder for spectrum dependent definitions
        # (adapt them in each class)
        # - (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = None
        # - (str) name you want to appear in the header of the
        #   file, see Cell.get_header
        self.hd_name = None
        # - (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def _get_raw_cl(self, cosmo, cl_name):
        """
        Convenience method to get unlensed Cls.
        Arguments:
        - cosmo: classy.Class() instance. We assume that
          the output was previously computed;
        - cl_name (str): type of cl with the same syntax as Class.
        """

        # Get cell
        sp = cosmo.raw_cl(lmax=self.ell_max)

        ell = sp['ell'][self.ell_min:]
        cl = sp[cl_name][self.ell_min:]

        cl *= ell*(ell+1.)/2./np.pi

        return cl

    def _get_lensed_cl(self, cosmo, cl_name):
        """
        Convenience method to get lensed Cls.
        Arguments:
        - cosmo: classy.Class() instance. We assume that
          the output was previously computed;
        - cl_name (str): type of cl with the same syntax as Class.
        """

        # Get cell
        sp = cosmo.lensed_cl(lmax=self.ell_max)

        ell = sp['ell'][self.ell_min:]
        cl = sp[cl_name][self.ell_min:]

        cl *= ell*(ell+1.)/2./np.pi

        return cl

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
        if self.ratio:
            hd = 'Ratio of the {} C_l for ell={} to {}.\n'
        else:
            hd ='dimensionless {} [l(l+1)/2pi] C_l for ell={} to {}.\n'
        # hd += '\t'.join(self.get_names())

        # Cl specific settings
        hd = hd.format(
            self.hd_name,
            self.ell_min,
            self.ell_max,
        )
        return hd


# ----------------- Pk -------------------------------------------------------#

class MatterPk(Pk):
    """
    Matter power spectrum.

    NOTE: k is in units of h/Mpc. P(k) is in units of (Mpc/h)^3.
    """

    def __init__(self, name, params):
        Pk.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'Total matter'
        return

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

    def __init__(self, name, params):
        Pk.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'CDM + baryons'
        return

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
        try:
            pk = np.array([cosmo.pk_cb(k, z_pk) for k in k_range])
        except classy.CosmoSevereError:
            pk = np.array([cosmo.pk(k, z_pk) for k in k_range])

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

    def __init__(self, name, params):
        Pk.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk', 'dTk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'Weyl'
        return

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


# ----------------- Cell -----------------------------------------------------#

class CellTT(Cell):
    """
    TT power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['tCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'tt'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'TT'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


class CellEE(Cell):
    """
    EE power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['pCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'ee'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'EE'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


class CellTE(Cell):
    """
    TE power spectrum.
    As in Class, we compute the dimensionless Cell using:

    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['tCl', 'pCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'te'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'TE'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


class CellBB(Cell):
    """
    BB power spectrum.
    As in Class, we compute the dimensionless Cell using:

    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['pCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'bb'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'BB'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


class Cellpp(Cell):
    """
    phi-phi power spectrum.
    As in Class, we compute the dimensionless Cell using:

    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'pp'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'phi-phi'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


class CellTp(Cell):
    """
    T-phi power spectrum.
    As in Class, we compute the dimensionless Cell using:

    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['tCl', 'lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'tp'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'T-phi'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = False
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_raw_cl(cosmo, self.class_name)


# ----------------- Cell - lensed --------------------------------------------#

class CellTTLensed(Cell):
    """
    TT lensed power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['tCl', 'lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'tt'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'TT lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)


class CellEELensed(Cell):
    """
    EE lensed power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['pCl', 'lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'ee'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'EE lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)


class CellTELensed(Cell):
    """
    TE lensed power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['tCl', 'pCl', 'lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'te'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'TE lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)


class CellBBLensed(Cell):
    """
    BB lensed power spectrum.
    As in Class, we compute the dimensionless Cell using:
    
    ell*(ell+1.)/2./pi * Cl

        """

    def __init__(self, name, params):
        Cell.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['pCl', 'lCl']
        # (str) name of the cl as in Class, i.e., tt, te, ee, ...
        self.class_name = 'bb'
        # (str) name you want to appear in the header of the
        # file, see Cell.get_header
        self.hd_name = 'BB lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)
