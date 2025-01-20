"""
.. module:: sampling_functions

:Synopsis: List of sampling functions.
:Author: Emilio Bellini

List of analytic functions that can be used to check
the performance of the emulator on known problems.
If you want to implement a new function do it here
and call it in the params file with the 'generate_sample'
variable.
"""

import numpy as np


# 1D functions

def linear_1d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x + b
    """
    a = params['a']
    b = params['b']
    x = x[x_names.index('x')]
    y = a*x + b
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


def quadratic_1d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x^2 + b*x + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x = x[x_names.index('x')]
    y = a*x**2 + b*x + c
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


def gaussian_1d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = exp(-(x-mean^2)/std/2)
    """
    mean = params['mean']
    std = params['std']
    x = x[x_names.index('x')]
    y = np.exp(-(x-mean)**2./std**2./2.)
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


# 2D functions

def linear_2d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x1 + b*x2 + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x1 = x[x_names.index('x1')]
    x2 = x[x_names.index('x2')]
    y = a*x1 + b*x2 + c
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


def quadratic_2d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e = params['e']
    f = params['f']
    x1 = x[x_names.index('x1')]
    x2 = x[x_names.index('x2')]
    y = a*x1**2. + b*x2**2. + c*x1*x2 + d*x1 + e*x2 + f
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


# Cobaya loglikelihoods

def cobaya_loglike(x, x_names, params, model=None, extra_args=None, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    """
    import cobaya
    # Merge dictionaries of params for cobaya
    if extra_args:
        cobaya_params = {'params': params} | extra_args
    else:
        cobaya_params = params
    # Get y_names
    y_names = list(cobaya_params['likelihood'].keys())
    # Define model
    if model is None:
        model = cobaya.model.get_model(cobaya_params)
    # Each sample should be a dictionary
    sampled_params = dict(zip(x_names, x))
    # Get loglike
    loglikes = model.loglikes(sampled_params, as_dict=True)[0]
    # Get y array
    y = np.array([loglikes[b] for b in y_names])
    # Add total loglike
    y = np.hstack((y, np.sum(y, axis=0)))
    y_names.append('tot_loglike')
    # Add total logprior
    y = np.hstack((y, model.logprior(sampled_params)))
    y_names.append('logprior')
    # Add total logposterior
    y = np.hstack((y, model.logpost(sampled_params)))
    y_names.append('logpost')
    # Replace nans with infinities
    y = np.nan_to_num(y, nan=-np.inf)
    y = y[np.newaxis]
    return y, y_names, None, model


# Class spectra

def class_spectra(x, x_names, params, extra_args=None, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    """
    # TODO: this is now working only for mPk (both total and cb)
    y = [
        np.array([[1, 2]]),
        np.array([[2, 3, 4]]),
    ]
    y_names = [
        ['x1', 'x2'],
        ['x2', 'x3', 'x4'],
    ]
    y_fnames = ['f1', 'f2']
    return y, y_names, y_fnames, None

    # Init classy
    import classy
    cosmo = classy.Class()

    # Build parameter dictionary to be computed
    model_params = dict(zip(x_names, x))
    try:
        model_params = model_params | params['extra_args']
    except KeyError:
        pass
    except TypeError:
        pass
    # Add the spectra wanted
    model_params['output'] = params['spectra']['names']
    model_params['P_k_max_h/Mpc'] = params['spectra']['k_max']

    # Get k_range
    k_range = np.logspace(
        np.log10(params['spectra']['k_min']),
        np.log10(params['spectra']['k_max']),
        num=params['spectra']['k_num']
    )
    y_names = ["k_h_Mpc_{}".format(k) for k in k_range]

    try:
        # Compute class
        cosmo.set(model_params)
        cosmo.compute()

        # Decide which PS is needed
        if params['spectra']['type_pk'] == 'tot':
            fun = cosmo.pk
        elif params['spectra']['type_pk'] == 'cb':
            fun = cosmo.pk_cb

        # it has to be in units of 1/Mpc
        k_range *= cosmo.h()

        # Get pk
        y = np.array([fun(k, model_params['z_pk']) for k in k_range])
        # The output is in units Mpc**3 and I want (Mpc/h)**3.
        y *= cosmo.h()**3.
    except classy.CosmoComputationError:
        # TODO: this has to be improved
        y = -1.*np.ones_like(k_range)

    return y, y_names, None, None
