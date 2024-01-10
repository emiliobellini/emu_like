"""
List of analytic functions that can be used to check
the performance of the emulator on known problems.
If you want to implement a new function do it here
and call it in the params file with the 'generate_sample'
variable.
"""
import numpy as np
import cobaya


# 1D functions

def linear_1d(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (1)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    y = a*x + b
    """
    a = params['a']
    b = params['b']
    x = x[x_var.index('x')]
    y = a*x + b
    y = y[np.newaxis]
    return y, None, None


def quadratic_1d(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (1)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    y = a*x^2 + b*x + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x = x[x_var.index('x')]
    y = a*x**2 + b*x + c
    y = y[np.newaxis]
    return y, None, None


def gaussian_1d(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (1)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    y = exp(-(x-mean^2)/std/2)
    """
    mean = params['mean']
    std = params['std']
    x = x[x_var.index('x')]
    y = np.exp(-(x-mean)**2./std**2./2.)
    y = y[np.newaxis]
    return y, None, None


# 2D functions

def linear_2d(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=2)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (1)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    y = a*x1 + b*x2 + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x1 = x[x_var.index('x1')]
    x2 = x[x_var.index('x2')]
    y = a*x1 + b*x2 + c
    y = y[np.newaxis]
    return y, None, None


def quadratic_2d(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=2)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (1)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e = params['e']
    f = params['f']
    x1 = x[x_var.index('x1')]
    x2 = x[x_var.index('x2')]
    y = a*x1**2. + b*x2**2. + c*x1*x2 + d*x1 + e*x2 + f
    y = y[np.newaxis]
    return y, None, None


# Cobaya loglikelihoods

def cobaya_loglike(x, x_var, params, model=None):
    """
    Arguments:
        x: array with dimensions (n_params=len(x_var))
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_likelihoods+3)
        y_names: list of names for each of the y column
        (it is possible to leave it None)

    """
    # Get y_names
    y_names = list(params['likelihood'].keys())
    # Define model
    if model is None:
        model = cobaya.model.get_model(params)
    # Each sample should be a dictionary
    sampled_params = dict(zip(x_var, x))
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
    return y, y_names, model
