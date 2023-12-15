"""
List of mcmc samplers.
"""
import emcee
import numpy as np
import sys
import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator
from tools.scalers import Scaler


class MCMC(object):

    def __init__(self, params, verbose=False):
        self.params = params[self.name]
        self.output = params['output']

        # Load params emulator
        emu_params = io.YamlFile(
            de.file_names['params']['name'],
            root=params['emulator']['path'],
            should_exist=True)
        emu_params.read()

        # Define emulator folder
        emu_folder = io.Folder(path=params['emulator']['path'])

        # Load scalers
        scalers = emu_folder.subfolder(de.file_names['x_scaler']['folder'])
        scaler_x_path = io.File(de.file_names['x_scaler']['name'],
                                root=scalers)
        scaler_y_path = io.File(de.file_names['y_scaler']['name'],
                                root=scalers)
        self.scaler_x = Scaler.load(scaler_x_path, verbose=verbose)
        self.scaler_y = Scaler.load(scaler_y_path, verbose=verbose)

        # Call the right emulator
        self.emu = Emulator.choose_one(
            emu_params, emu_folder, verbose=verbose)

        # Load emulator
        self.emu.load(
            model_to_load=params['emulator']['epoch'],
            verbose=verbose)

        # Load sample details
        sample_path = emu_folder.subfolder(
            de.file_names['sample_details']['folder'])
        self.sample_details = io.YamlFile(
            de.file_names['sample_details']['name'],
            root=sample_path)
        self.sample_details.read()

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

    def evaluate_emulator(self, x, model, scaler_x, scaler_y):
        x_reshaped = np.array([x])
        if scaler_x:
            x_scaled = scaler_x.transform(x_reshaped)
        else:
            x_scaled = x_reshaped
        y_scaled = model(x_scaled, training=False)
        y = scaler_y.inverse_transform(y_scaled)
        return y

    def log_prior(self, x, x_names, bounds):
        for pos, name in enumerate(x_names):
            if (x[pos] < bounds[name][0]) or (x[pos] > bounds[name][1]):
                return -np.inf
        return 0.0

    def log_prob(self, x, model, x_names, bounds, scaler_x, scaler_y):
        log_lkl = self.evaluate_emulator(x, model, scaler_x, scaler_y)[0, 0]
        return -0.5*log_lkl + self.log_prior(x, x_names, bounds)

    def run(self):
        return


class EmceeMCMC(MCMC):

    def __init__(self, params, verbose):
        if verbose:
            scp.info('Initializing EmceeMCMC sampler.')
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
                self.scaler_x,
                self.scaler_y],
            threads=n_threads)

        # Initial positions
        bounds = np.array([self.sample_details['bounds'][x]
                           for x in self.sample_details['x_names']])
        center = np.mean(bounds, axis=1)
        width = np.array([x[1]-x[0] for x in bounds])
        self.pos = center + width*squeeze_factor*np.random.randn(
            n_walkers, n_dim)

        # Header
        header = '# weight\t-logprob\t'+'\t'.join(
            self.sample_details['x_names'])+'\n'

        self.chains = io.File(
            de.file_names['chains']['name'],
            root=params['output']).create(header=header, verbose=verbose)
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
        return

    def run(self):
        return
