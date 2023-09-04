import emcee
import numpy as np
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

    sampler.run_mcmc(pos, n_steps, progress=True)

    return
