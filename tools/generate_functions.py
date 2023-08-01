"""
List of analytic functions that can be used to check
the performance of the emulator on known problems.
If you want to implement a new function do it here
and call it in the params file with the 'generate_sample'
variable.
"""
import numpy as np


# 1D functions

def linear_1d(x, x_var, params):
    """
    Arguments:
        x: array with dimensions (n_samples, 1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_samples, 1)

    y = a*x + b
    """
    a = params['a']
    b = params['b']
    x = x[:, x_var.index('x')]
    y = a*x + b
    y = y[:, np.newaxis]
    return y


def quadratic_1d(x, x_var, params):
    """
    Arguments:
        x: array with dimensions (n_samples, 1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_samples, 1)

    y = a*x^2 + b*x + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x = x[:, x_var.index('x')]
    y = a*x**2 + b*x + c
    y = y[:, np.newaxis]
    return y


def gaussian_1d(x, x_var, params):
    """
    Arguments:
        x: array with dimensions (n_samples, 1)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_samples, 1)

    y = exp(-(x-mean^2)/std/2)
    """
    mean = params['mean']
    std = params['std']
    x = x[:, x_var.index('x')]
    y = np.exp(-(x-mean)**2./std**2./2.)
    y = y[:, np.newaxis]
    return y


# 2D functions

def linear_2d(x, x_var, params):
    """
    Arguments:
        x: array with dimensions (n_samples, 2)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_samples, 1)

    y = a*x1 + b*x2 + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x1 = x[:, x_var.index('x1')]
    x2 = x[:, x_var.index('x2')]
    y = a*x1 + b*x2 + c
    y = y[:, np.newaxis]
    return y


def quadratic_2d(x, x_var, params):
    """
    Arguments:
        x: array with dimensions (n_samples, 2)
        x_var: list of varying parameters names
        params: dictionary of all parameters
    Output:
        y: array with dimensions (n_samples, 1)

    y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e = params['e']
    f = params['f']
    x1 = x[:, x_var.index('x1')]
    x2 = x[:, x_var.index('x2')]
    y = a*x1**2. + b*x2**2. + c*x1*x2 + d*x1 + e*x2 + f
    y = y[:, np.newaxis]
    return y
