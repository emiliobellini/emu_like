"""
.. module:: plots

:Synopsis: Module managing plots.
:Author: Emilio Bellini
TODO: improve it
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from . import io as io


class SinglePlot(object):

    def __init__(self, coords, labels, x_label, y_label, root,
                 fname=None, verbose=False):
        self.labels = labels
        self.x_label = x_label
        self.y_label = y_label
        self.root = root.create(verbose=verbose)
        self.coords = [self._sort_coordinates(x, y) for x, y in coords]
        self.verbose = verbose
        self.fname = fname
        # TODO: if we realise we need more linestyles, just add them here
        self.lines = ['-']

        for nc in range(len(self.coords)):
            nl = np.mod(nc, len(self.lines))
            x = self.coords[nc][0]
            y = self.coords[nc][1]
            lab = self.labels[nc]
            self.plot(x, y, lab, self.lines[nl])
        self.decorate()
        return

    def _sort_coordinates(self, x, y):
        npx = np.array(x)
        npy = np.array(y)
        idx = np.argsort(npx)
        return (npx[idx], npy[idx])

    def decorate(self):
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        return

    def plot(self, x, y, lab, ls):
        plt.plot(x, y, ls, label=lab, lw=1)
        return self

    def save(self):
        if self.fname:
            name = self.fname
        else:
            name = '{}_vs_{}.pdf'.format(self.x_label, self.y_label)
        fpath = os.path.join(self.root, name)
        plt.savefig(fpath)
        plt.close()
        if self.verbose:
            io.print_level(1, 'Saved plot at {}'.format(fpath))
        return


class ScatterPlot(SinglePlot):

    def __init__(self, *args, **kwargs):
        SinglePlot.__init__(self, *args, **kwargs)
        return

    def plot(self, x, y, lab, ls):
        plt.scatter(x, y, label=lab, s=1)
        return self


class LogLogPlot(SinglePlot):

    def __init__(self, *args, **kwargs):
        SinglePlot.__init__(self, *args, **kwargs)
        return

    def plot(self, x, y, lab, ls):
        plt.loglog(x, y, ls, label=lab, lw=1)
        return self
