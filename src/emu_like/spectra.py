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

    def __init__(self, dict_or_list):
        """
        Init the list of spectra to be computed.
        Arguments:
        - dict_or_list (dict or list): either nested dictionary
          of parameters for each spectrum, or a list of Spectrum objects.
        """
        # Init classy
        if isinstance(dict_or_list, dict):
            self.list = [Spectrum.choose_one(sp, dict_or_list[sp]) for sp in dict_or_list]
        elif isinstance(dict_or_list, list):
            self.list = dict_or_list
        
        # List of names of the spectra
        self.names = self.get_names()
        return

    def __setitem__(self, item, value):
        """
        Make Spectra subsriptable. Item can be:
        - integer: position of the spectrum
        - string: name of the spectrum
        In both cases it assigns value to the corresponding Spectrum.
        """
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str):
            idx = self._get_idx_from_name(item)
        self.list[idx] = value

    def __getitem__(self, item):
        """
        Make Spectra subsriptable. Item can be:
        - integer: position of the spectrum
        - string: name of the spectrum
        In both cases it returns the corresponding Spectrum object.
        """
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str):
            idx = self._get_idx_from_name(item)
        return self.list[idx]

    def _get_idx_from_name(self, name):
        """
        From the name (str) of the Spectrum,
        it returns its position (int) in Spectra.
        """
        idx = self.names.index(name)
        return idx

    def _get_name_from_idx(self, idx):
        """
        From the position (int) of the Spectrum,
        it returns its name (str).
        """
        return self.list[idx].name

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

        # ell max Cell
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
        Get a list of the spectra names.
        """
        names = [sp.name for sp in self.list]
        return names

    def get_y_names(self):
        """
        Get a list of the y names of each spectrum computed.
        """
        names = [sp.get_y_names() for sp in self.list]
        return names

    def get_headers(self):
        """
        Get a list of the y headers of each spectrum computed.
        """
        headers = [sp.get_header() for sp in self.list]
        return headers


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
        # Growth rates
        elif spectrum_type == 'fk_m':
            return MatterGrowthRate(spectrum_type, params)
        elif spectrum_type == 'fk_cb':
            return ColdBaryonGrowthRate(spectrum_type, params)
        elif spectrum_type == 'fk_weyl':
            return WeylGrowthRate(spectrum_type, params)
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
        elif spectrum_type == 'cl_pp_lensed':
            return CellppLensed(spectrum_type, params)
        elif spectrum_type == 'cl_Tp_lensed':
            return CellTpLensed(spectrum_type, params)
        else:
            raise ValueError(
                'Spectrum {} not recognized!'.format(spectrum_type))

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

    def _get_2D_pk(self, cosmo, k_range, only_cb):
        """
        Here the k_range is expected to be in units of 1/Mpc
        and pk is in units of Mpc^3.
        """

        # Decide if non linear
        if 'non_linear' in cosmo.pars:
            nonlinear = True
        else:
            nonlinear = False

        # Get array of pk
        pk_array, k_array, z_array = cosmo.get_pk_and_k_and_z(
            nonlinear=nonlinear,
            only_clustering_species = only_cb,
            h_units=False)

        # Flip z_array (for the interpolation it has to be increasing)
        z_array = np.flip(z_array)
        pk_array = np.flip(pk_array, axis=1)

        # Evaluate pk at the requested range
        pk = interp.make_splrep(k_array, pk_array, s=0)(k_range)

        return pk, z_array

    def get_n_vec(self):
        """
        Return the size of the data vector.
        """
        return self.k_num

    def get_y_names(self):
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
        # hd += '\t'.join(self.get_y_names())

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

    def get_y_names(self):
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
        # hd += '\t'.join(self.get_y_names())

        # Cl specific settings
        hd = hd.format(
            self.hd_name,
            self.ell_min,
            self.ell_max,
        )
        return hd


class GrowthRate(Pk):
    """
    Base class for the scale dependent growth rate f.

    Return the scale dependent growth factor
    f(z)= 1/2 * [d ln P(k,a) / d ln a]
        = - 0.5 * (1+z) * [d ln P(k,z) / d z]
    where P(k,z) is the power spectrum used

    NOTE: k is in units of h/Mpc. f(k) is dimensionless.
    """

    def __init__(self, name, params):
        Pk.__init__(self, name, params)
        return

    def get_header(self):
        """
        Header for the growth rate.
        """
        if self.ratio:
            hd = 'Ratio of the {} growth rate f(k) w.r.t. the reference f(k) '
        else:
            hd = '{} growth rate f(k) '
        hd += 'as a function of k (h/Mpc).\nk_min (h/Mpc) = {}, '
        hd += 'k_max (h/Mpc) = {}, {}-sampled, for a total number '
        hd += 'of k_modes of {}.\n'
        # hd += '\t'.join(self.get_y_names())

        hd = hd.format(
            self.hd_name,
            self.k_min,
            self.k_max,
            self.k_space,
            self.k_num
        )
        return hd

    def get(self, cosmo, z=None):
        """
        Get the growth rate of the desired spectrum.
        This is the same for each spectrum, provided that
        self.pk points to the write one (definition in __init__)
        """

        # Get array of pk
        pk_array = self.pk.get(cosmo, z=None)

        # If z is None return f(k, z)
        if z is None:
            # Compute pk
            pk = pk_array
            # Compute derivative (d ln P / d ln z)
            dpkdz = interp.make_splrep(
                self.pk.z_array, pk_array.T, s=0).derivative()(self.pk.z_array).T
            # Compute growth factor f
            fk = -0.5 * (1+self.pk.z_array) * dpkdz/pk
            # Store the z_array
            self.z_array = self.pk.z_array

        # Otherwise return f(k)
        else:
            # Compute pk
            pk = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z)
            # Compute derivative (d ln P / d ln z)
            if True:
                dpkdz = interp.make_splrep(
                    self.pk.z_array, pk_array.T, s=0).derivative()(z)
            # Here we keep also the manual derivative because the growth rate is noisy
            # and we may want to check it is less noisy with this (for now they are equivalent)
            else:
                z_step = 0.1
                if z - z_step >= 0.:
                    pk_p1 = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z+z_step)
                    pk_m1 = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z-z_step)
                    dpkdz = (pk_p1-pk_m1)/(2.*z_step)
                elif z - z_step/10 >= 0.:
                    z_step = z
                    pk_p1 = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z+z_step)
                    pk_m1 = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z-z_step)
                    dpkdz = (pk_p1-pk_m1)/(2.*z_step)
                else:
                    z_step /=10
                    pk_p1 = interp.make_splrep(self.pk.z_array, pk_array.T, s=0)(z+z_step)
                    dpkdz = (pk_p1-pk)/z_step
            # Compute growth factor f
            fk = -0.5 * (1+z) * dpkdz/pk

        return fk


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

    def get(self, cosmo, z=None):
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # If z is None return P(k, z)
        if z is None:
            pk, z_array = self._get_2D_pk(cosmo, k_range, only_cb=False)
            # Store the z_array
            self.z_array = z_array

        # Otherwise return P(k)
        else:
            pk = np.array([cosmo.pk(k, z) for k in k_range])

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

    def get(self, cosmo, z=None):
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # If z is None return P(k, z)
        if z is None:
            pk, z_array = self._get_2D_pk(cosmo, k_range, only_cb=True)
            # Store the z_array
            self.z_array = z_array

        # Otherwise return P(k)
        else:
            try:
                pk = np.array([cosmo.pk_cb(k, z) for k in k_range])
            except classy.CosmoSevereError:
                pk = np.array([cosmo.pk(k, z) for k in k_range])

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

    def get(self, cosmo, z=None):
        """
        Return the correct spectrum sampled at k_range bins.
        """

        # convert k in units of 1/Mpc
        k_range = self.k_range * cosmo.h()

        # Decide if non linear
        if 'non_linear' in cosmo.pars:
            nonlinear = True
        else:
            nonlinear = False

        # Get array of pk
        pk_array, k_array, z_array = cosmo.get_Weyl_pk_and_k_and_z(
            nonlinear=nonlinear,
            h_units=False)

        # Flip z_array (for the interpolation it has to be increasing)
        z_array = np.flip(z_array)
        pk_array = np.flip(pk_array, axis=1)

        # Evaluate pk at the requested range
        pk = interp.make_splrep(k_array, pk_array, s=0)(k_range)

        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        pk *= cosmo.h()**3.

        # Store the z_array
        if z is None:
            self.z_array = z_array
        # Or interpolate and get pk at the correct z
        else:
            pk = interp.make_splrep(z_array, pk.T, s=0)(z)

        return pk


class MatterGrowthRate(GrowthRate):
    """
    Scale dependent total matter growth rate f.

    Return the scale dependent growth factor
    f(z)= 1/2 * [d ln P(k,a) / d ln a]
        = - 0.5 * (1+z) * [d ln P(k,z) / d z]
    where P(k,z) is the total matter power spectrum

    NOTE: k is in units of h/Mpc. f(k) is dimensionless.
    """

    def __init__(self, name, params):
        GrowthRate.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'Total matter'
        # Define matter pk object
        self.pk = MatterPk(name='pk_m', params=self.params)
        return


class ColdBaryonGrowthRate(GrowthRate):
    """
    Scale dependent CDM+baryon growth rate f.

    Return the scale dependent growth factor
    f(z)= 1/2 * [d ln P(k,a) / d ln a]
        = - 0.5 * (1+z) * [d ln P(k,z) / d z]
    where P(k,z) is the CDM+baryon power spectrum

    NOTE: k is in units of h/Mpc. f(k) is dimensionless.
    """

    def __init__(self, name, params):
        GrowthRate.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'CDM + baryons'
        # Define matter pk object
        self.pk = ColdBaryonPk(name='pk_cb', params=self.params)
        return


class WeylGrowthRate(GrowthRate):
    """
    Scale dependent Weyl growth rate f.

    Return the scale dependent growth factor
    f(z)= 1/2 * [d ln P(k,a) / d ln a]
        = - 0.5 * (1+z) * [d ln P(k,z) / d z]
    where P(k,z) is the Weyl power spectrum

    NOTE: k is in units of h/Mpc. f(k) is dimensionless.
    """

    def __init__(self, name, params):
        GrowthRate.__init__(self, name, params)

        # (list of str) list of spectra that Class should compute.
        # Use the same syntax of the Class output argument.
        self.class_spectra = ['mPk', 'dTk']
        # (str) name you want to appear in the header of the
        # file, see Pk.get_header
        self.hd_name = 'Weyl'
        # Define matter pk object
        self.pk = WeylPk(name='pk_weyl', params=self.params)
        return


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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
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

    def get(self, cosmo, **kwargs):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)


class CellppLensed(Cell):
    """
    Tp lensed power spectrum.
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
        self.hd_name = 'phi-phi lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo, **kwargs):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)


class CellTpLensed(Cell):
    """
    Tp lensed power spectrum.
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
        self.hd_name = 'T-phi lensed'
        # (bool) True if the spectrum needs lensing
        self.want_lensing = True
        return

    def get(self, cosmo, **kwargs):
        """
        Return the correct spectrum sampled up to ell max.
        """
        return self._get_lensed_cl(cosmo, self.class_name)
