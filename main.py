"""
.. module:: emu_like

:Synopsis: Main pipeline
:Author: Emilio Bellini

This just redirects to the requested pipeline:
- pipelines.sample.sample_emu: generate a sample
- pipelines.train.train_emu: train an emulator
- pipelines.mcmc.mcmc_emu: run mcmc with an emulator
"""

import sys
from emu_like.io import argument_parser
from emu_like.y_models import YModel


# Define these two functions at the top level for parallel sampling
def init_parallel_sampling(y_name, params, y_outputs, n_samples, y_args, verbose=False):
    if verbose:
        print('Started init parallel sampling process')
    global y_model
    y_model = YModel.choose_one(
        y_name,
        params,
        y_outputs,
        n_samples,
        **y_args)
    if verbose:
        print('----> Finished init parallel sampling process')
    pass

def do_parallel_sampling(x, nx, verbose=False):
    if verbose:
        print('Started parallel sampling process')
    result = y_model.evaluate(x, nx)
    if verbose:
        print('----> Finished parallel sampling process')
    return result


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = argument_parser()

    # Redirect the run to the correct module
    if args.mode == 'sample':
        from pipelines.sample import sample_emu
        sys.exit(sample_emu(args, init_parallel_sampling, do_parallel_sampling))
    if args.mode == 'train':
        from pipelines.train import train_emu
        sys.exit(train_emu(args))
    if args.mode == 'mcmc':
        from pipelines.mcmc import mcmc_emu
        sys.exit(mcmc_emu(args))
    if args.mode == 'export':
        from pipelines.export import export_emu
        sys.exit(export_emu(args))
