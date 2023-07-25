"""
List of analytic functions that can be used to check
the performance of the emulator on known problems.
If you want to implement a new function do it here
and call it in the params file with the 'generate_sample'
variable.
"""
import numpy as np


# 1D functions

def linear_1d(x, a=1., b=0.):
    return a*x + b


def quadratic_1d(x, a=1., b=0., c=0.):
    return a*x**2 + b*x + c


def gaussian_1d(x, mean=1., std=0.):
    return np.exp(-(x-mean)**2./std**2./2.)


# 2D functions

def linear_2d(x, a=[1., 1.], b=0.):
    return a[0]*x[:, 0] + a[1]*x[:, 1] + b


def quadratic_2d(x, a=[1., 0., 1.], b=[0., 0.], c=0.):
    """
    The kewords arguments are vectors.
    The function becomes:
        f(x1, x2) = a0*x1**2 + a1*x1*x2 + a2*x2**2 + b0*x1 + b1*x2 + c0
    """
    f = a[0]*x[:, 0]**2 + a[1]*x[:, 0]*x[:, 1] + a[2]*x[:, 1]**2.
    f += b[0]*x[:, 0] + a[1]*x[:, 1]
    f += c[0]
    return f[:, np.newaxis]
