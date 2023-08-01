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

    def get_x(self, params, varying, n_samples):
        mins = [params[x]['min'] for x in varying]
        maxs = [params[x]['max'] for x in varying]
        x = np.linspace(mins, maxs, num=n_samples)
        return x


class LogGridSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing LogGrid sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, params, varying, n_samples):
        mins = [np.log10(params[x]['min']) for x in varying]
        maxs = [np.log10(params[x]['max']) for x in varying]
        x = np.logspace(mins, maxs, num=n_samples)
        return x


class RandomUniformSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing RandomUniform sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, params, varying, n_samples):
        mins = [params[x]['min'] for x in varying]
        maxs = [params[x]['max'] for x in varying]
        x = np.random.uniform(mins, maxs, size=(n_samples, len(mins)))
        return x


class RandomNormalSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing RandomNormal sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, params, varying, n_samples):
        means = [params[x]['loc'] for x in varying]
        std = [params[x]['scale'] for x in varying]
        x = np.random.normal(means, std, size=(n_samples, len(means)))
        return x


class LatinHypercubeSampler(Sampler):

    def __init__(self, verbose=False):
        if verbose:
            scp.info('Initializing LatinHypercube sampler.')
        Sampler.__init__(self)
        return

    def get_x(self, params, varying, n_samples):
        mins = [params[x]['min'] for x in varying]
        maxs = [params[x]['max'] for x in varying]
        sampler = scipy.stats.qmc.LatinHypercube(d=len(mins))
        sample = sampler.random(n=n_samples)
        x = scipy.stats.qmc.scale(sample, mins, maxs)
        return x
