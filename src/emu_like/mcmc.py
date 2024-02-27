"""
.. module:: mcmc

:Synopsis: Module containing list of mcmc samplers.
:Author: Emilio Bellini
"""

import emcee
import numpy as np
import os
import sys
from . import defaults as de
from . import io as io
from .emu import Emulator


class MCMC(object):

    def __init__(self, params, verbose=False):
        """
        Generic initialisation for every MCMC sampler.
        Stored attributes:
        - give the correct name to the sampler;
        - load the emulator;
        - fix output path;
        - parameters to be varied;
        - create output folder.
        """

        # Output path
        self.output = params['output']

        if io.Folder(self.output).is_empty():
            if verbose:
                io.info("Writing output in {}".format(self.output))
        else:
            raise Exception(
                'Output folder not empty! Exiting to avoid corruption of '
                'precious data!')

        # Create chains folder
        io.Folder(self.output).create(verbose=verbose)

        self.name = list(params['sampler'].keys())[0]

        # Load emulator
        self.emu = Emulator().load(
            params['emulator']['path'],
            model_to_load=params['emulator']['epoch'],
            verbose=verbose)

        # Varied parameters
        self.params = params['params']

        return

    @staticmethod
    def choose_one(params, verbose=False):
        """
        Main function to get the correct MCMC sampler.
        Arguments:
        - params (dict): dictionary where the only key
          is the type of sampler and the corresponding
          value is a nested dictionary with the parameters
          needed by the sampler;
        - verbose (bool, default: False): verbosity.
        Return:
        - MCMC (object): based on params, get the correct
          sampler and initialise it.
        """
        if len(params['sampler'].keys()) > 1:
            raise Exception('Multiple samplers! Choose one!')
        elif len(params['sampler'].keys()) == 0:
            raise Exception('No sampler specified! Choose one!')
        else:
            if 'emcee' in params['sampler'].keys():
                return EmceeMCMC(params, verbose)
            else:
                raise ValueError('MCMC Sampler not recognized!')

    def log_prior(self, x, bounds):
        """
        Deal with priors.
        For now we implemented only flat priors.
        Arguments:
        - x (list or array): array of values for input parameters;
        - bounds (list or array): list of [min, max] for each parameter.
        """
        for pos, _ in enumerate(x):
            if (x[pos] < bounds[pos][0]) or (x[pos] > bounds[pos][1]):
                return -np.inf
        return 0.0

    def log_prob(self, x, bounds):
        """
        Evaluate the log likelihood from the emulator
        and combine this with log_prior to get the
        log posterior.
        Arguments:
        - x (list or array): array of values for input parameters;
        - bounds (list or array): list of [min, max] for each parameter.
        """
        log_lkl = self.emu.eval(x)
        return -0.5*log_lkl + self.log_prior(x, bounds)


class EmceeMCMC(MCMC):

    def __init__(self, params, verbose=False):
        """
        Initialise Emcee.
        The main tasks are:
        - initialise the sampler with the parameters specified;
        - get initial positions for each walker;
        - create chains file.
        """
        if verbose:
            io.info('Initializing EmceeMCMC sampler.')

        MCMC.__init__(self, params, verbose=verbose)

        # Get bounds
        bounds = [[self.params[x]['prior']['min'],
                   self.params[x]['prior']['max']] for x in self.emu.x_names]

        # Define emcee parameters
        n_walkers = params['sampler'][self.name]['n_walkers']
        n_threads = params['sampler'][self.name]['n_threads']
        squeeze_factor = params['sampler'][self.name]['squeeze_factor']
        self.n_steps = params['sampler'][self.name]['n_steps']
        n_dim = len(self.emu.x_names)

        # Init sampler
        self.sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_prob,
            args=[bounds],
            threads=n_threads)

        # Initial positions
        center = np.mean(bounds, axis=1)
        width = np.array([x[1]-x[0] for x in bounds])
        self.pos = center + width*squeeze_factor*np.random.randn(
            n_walkers, n_dim)

        # Header chains file
        header = '# weight\t-logprob\t'+'\t'.join(self.emu.x_names)+'\n'

        # Create chains file
        self.chains_path = os.path.join(
            self.output, de.file_names['chains']['name'])
        with open(self.chains_path, 'w') as fn:
            fn.write(header)
            if verbose:
                io.info('Writing chains at {}'.format(self.chains_path))

        return

    def run(self):
        """
        Run the sampler starting from self.pos
        with the specified settings.
        """
        for count, result in enumerate(self.sampler.sample(
                self.pos, iterations=self.n_steps)):
            x_vars = result[0]
            prob = result[1]
            f = open(self.chains_path, 'a')
            for k in range(self.pos.shape[0]):
                out = np.append(np.array([1., -prob[k]]), x_vars[k])
                f.write('    '.join(
                    ['{0:.10e}'.format(x) for x in out]) + '\n')
            f.close()
            if np.mod(count, 10) == 0:
                print('----> Computed {0:5.1%} of the steps'
                      ''.format(float(count+1) / self.n_steps))
            sys.stdout.flush()

        return
