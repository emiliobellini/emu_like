import emcee
import numpy as np
import sys
import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator
from tools.scalers import Scaler


def evaluate_emulator(x, model, scaler_x, scaler_y):
    x_reshaped = np.array([x])
    if scaler_x:
        x_scaled = scaler_x.transform(x_reshaped)
    else:
        x_scaled = x_reshaped
    y_scaled = model(x_scaled, training=False)
    y = scaler_y.inverse_transform(y_scaled)
    return y


def log_prior(x, x_names, bounds):
    for pos, name in enumerate(x_names):
        if (x[pos] < bounds[name][0]) or (x[pos] > bounds[name][1]):
            return -np.inf
    return 0.0


def log_prob(x, model, x_names, bounds, scaler_x, scaler_y):
    log_lkl = evaluate_emulator(x, model, scaler_x, scaler_y)[0, 0]
    return -0.5*log_lkl + log_prior(x, x_names, bounds)


def test_mcmc_emu(args):
    """ Test the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        scp.print_level(0, "\nStarted testing emulated mcmc\n")

    # Load input file
    params = io.YamlFile(args.params_file, should_exist=True)
    params.read()

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
    scaler_x_path = io.File(de.file_names['x_scaler']['name'], root=scalers)
    scaler_y_path = io.File(de.file_names['y_scaler']['name'], root=scalers)
    scaler_x = Scaler.load(scaler_x_path, verbose=args.verbose)
    scaler_y = Scaler.load(scaler_y_path, verbose=args.verbose)

    # Call the right emulator
    emu = Emulator.choose_one(emu_params, emu_folder, verbose=args.verbose)

    # Load emulator
    emu.load(model_to_load=params['emulator']['epoch'], verbose=args.verbose)

    # Load sample details
    sample_path = emu_folder.subfolder(
        de.file_names['sample_details']['folder'])
    sample_details = io.YamlFile(de.file_names['sample_details']['name'],
                                 root=sample_path)
    sample_details.read()

    # Define emcee parameters
    n_walkers = params['emcee']['n_walkers']
    n_dim = len(sample_details['x_names'])
    n_threads = params['emcee']['n_threads']
    n_steps = params['emcee']['n_steps']
    squeeze_factor = params['emcee']['squeeze_factor']

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_prob,
        args=[
            emu.model,
            sample_details['x_names'],
            sample_details['bounds'],
            scaler_x,
            scaler_y],
        threads=n_threads)

    # Initial positions
    bounds = np.array([sample_details['bounds'][x]
                       for x in sample_details['x_names']])
    center = np.mean(bounds, axis=1)
    width = np.array([x[1]-x[0] for x in bounds])
    pos = center + width*squeeze_factor*np.random.randn(n_walkers, n_dim)

    # Header
    header = '# weight\t-logprob\t'+'\t'.join(sample_details['x_names'])+'\n'

    chains = io.File(de.file_names['chains']['name'],
                     root=params['output']).create(header=header,
                                                   verbose=args.verbose)
    for count, result in enumerate(sampler.sample(pos, iterations=n_steps)):
        x_vars = result[0]
        prob = result[1]
        f = open(chains.path, 'a')
        for k in range(pos.shape[0]):
            out = np.append(np.array([1., -prob[k]]), x_vars[k])
            f.write('    '.join(['{0:.10e}'.format(x) for x in out]) + '\n')
        f.close()
        if np.mod(count, 10) == 0:
            print('----> Computed {0:5.1%} of the steps'
                  ''.format(float(count+1) / n_steps))
        sys.stdout.flush()

    return
