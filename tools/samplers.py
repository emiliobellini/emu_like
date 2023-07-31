import numpy as np
import scipy
import tools.printing_scripts as scp


class Sampler(object):

    def __init__(self):
        return

    @staticmethod
    def choose_one(sampler_type, verbose=False):
        """
        Main function to get the correct Emulator.

        Arguments:
            - sampler_type (str): type of sampler.

        Return:
            - Emulator (object): based on params, get the correct
              emulator and initialize it.
        """
        if sampler_type == 'grid':
            return GridSampler(verbose=verbose)
        elif sampler_type == 'log_grid':
            return LogGridSampler(verbose=verbose)
        elif sampler_type == 'random_uniform':
            return RandomUniformSampler(verbose=verbose)
        elif sampler_type == 'random_normal':
            return RandomNormalSampler(verbose=verbose)
        elif sampler_type == 'latin_hypercube':
            return LatinHypercubeSampler(verbose=verbose)
        else:
            raise ValueError('Sampler not recognized!')

    def get_x(self, bounds):
        """
        Placeholder for get_x
        """
        return


class GridSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing Grid sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, bounds, n_samples):
        x = np.linspace(bounds[:, 0], bounds[:, 1], num=n_samples)
        return x


class LogGridSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing LogGrid sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, bounds, n_samples):
        lefts, rights = np.log10(bounds[:, 0]), np.log10(bounds[:, 1])
        x = np.logspace(lefts, rights, num=self.n_samples)
        return x


class RandomUniformSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing RandomUniform sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, bounds, n_samples):
        x = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(n_samples, len(bounds[:, 0])))
        return x


class RandomNormalSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing RandomNormal sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, bounds, n_samples):
        x = np.random.normal(
            bounds[:, 0], bounds[:, 1],
            size=(n_samples, len(bounds[:, 0])))
        return x


class LatinHypercubeSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing LatinHypercube sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, bounds, n_samples):
        sampler = scipy.stats.qmc.LatinHypercube(d=len(bounds[:, 0]))
        sample = sampler.random(n=n_samples)
        x = scipy.stats.qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        return x
