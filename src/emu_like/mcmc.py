"""
.. module:: mcmc

:Synopsis: Module containing list of mcmc samplers.
:Author: Emilio Bellini
TODO: this does not work
"""

import cobaya
# from cobaya.run import run as run_cob
import emcee
import numpy as np
import sys
# from . import defaults as de
from . import io as io
from .emu import Emulator
# from .params import Params


class MCMC(object):

    def __init__(self, params, verbose=False):
        self.params = params[self.name]
        self.output = params['output']

        # Load params emulator
        # emu_params = Params().load(
        #     de.file_names['params']['name'],
        #     root=params['emulator']['path'])

        # Call the right emulator
        self.emu = Emulator.choose_one(params['emulator']['type'],
                                       verbose=verbose)
        # Load emulator
        self.emu.load(params['output'], model_to_load='best', verbose=verbose)

        return

    @staticmethod
    def choose_one(params, verbose=False):
        # Get right sampler
        if 'cobaya' in params.keys():
            return CobayaMCMC(params, verbose)
        elif 'emcee' in params.keys():
            return EmceeMCMC(params, verbose)
        else:
            raise ValueError('MCMC Sampler not recognized!')

    def evaluate_emulator(self, x, model, x_scaler, y_scaler):
        x_reshaped = np.array([x])
        if x_scaler:
            x_scaled = x_scaler.transform(x_reshaped)
        else:
            x_scaled = x_reshaped
        y_scaled = model(x_scaled, training=False)
        y = y_scaler.inverse_transform(y_scaled)
        return y

    def log_prior(self, x, x_names, bounds):
        for pos, name in enumerate(x_names):
            if (x[pos] < bounds[name][0]) or (x[pos] > bounds[name][1]):
                return -np.inf
        return 0.0

    def log_prob(self, x, model, x_names, bounds, x_scaler, y_scaler):
        log_lkl = self.evaluate_emulator(x, model, x_scaler, y_scaler)[0, 0]
        return -0.5*log_lkl + self.log_prior(x, x_names, bounds)

    def run(self):
        return


class EmceeMCMC(MCMC):

    def __init__(self, params, verbose):
        if verbose:
            io.info('Initializing EmceeMCMC sampler.')
        self.name = 'emcee'
        MCMC.__init__(self, params, verbose)

        # Define emcee parameters
        n_walkers = self.params['n_walkers']
        n_dim = len(self.sample_details['x_names'])
        n_threads = self.params['n_threads']
        squeeze_factor = self.params['squeeze_factor']

        self.sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_prob,
            args=[
                self.emu.model,
                self.sample_details['x_names'],
                self.sample_details['bounds'],
                self.x_scaler,
                self.y_scaler],
            threads=n_threads)

        # Initial positions
        bounds = np.array([self.sample_details['bounds'][x]
                           for x in self.sample_details['x_names']])
        center = np.mean(bounds, axis=1)
        width = np.array([x[1]-x[0] for x in bounds])
        self.pos = center + width*squeeze_factor*np.random.randn(
            n_walkers, n_dim)

        # Header
        # header = '# weight\t-logprob\t'+'\t'.join(
        #     self.sample_details['x_names'])+'\n'

        # Create chains file
        return

    def run(self):

        n_steps = self.params['n_steps']
        for count, result in enumerate(self.sampler.sample(
                self.pos, iterations=n_steps)):
            x_vars = result[0]
            prob = result[1]
            f = open(self.chains.path, 'a')
            for k in range(self.pos.shape[0]):
                out = np.append(np.array([1., -prob[k]]), x_vars[k])
                f.write('    '.join(
                    ['{0:.10e}'.format(x) for x in out]) + '\n')
            f.close()
            if np.mod(count, 10) == 0:
                print('----> Computed {0:5.1%} of the steps'
                      ''.format(float(count+1) / n_steps))
            sys.stdout.flush()

        return


class CobayaMCMC(MCMC):

    def __init__(self, params, verbose):
        if verbose:
            io.info('Initializing CobayaMCMC sampler.')
        self.name = 'cobaya'
        MCMC.__init__(self, params, verbose)

        # Build info dict
        self.info = {
            'output': self.output,
            'sampler': {'mcmc': self.params},
            'params': params['params'],
            'likelihood': {
                'like': self.log_like
            },
        }

        return

    def log_like(self, *params):
        # TODO: this is not working, it does not accept unnamed parameters
        # Probably implement a likelihood class
        log_lkl = self.evaluate_emulator(
            params, self.emu.model, self.x_scaler, self.y_scaler)[0, 0]
        print(log_lkl)
        return log_lkl, {}

    def run(self):
        updated_info, sampler = cobaya.run(self.info)
        return
