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


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = argument_parser()

    # Redirect the run to the correct module
    if args.mode == 'sample':
        from pipelines.sample import sample_emu
        sys.exit(sample_emu(args))
    if args.mode == 'train':
        from pipelines.train import train_emu
        sys.exit(train_emu(args))
    if args.mode == 'mcmc':
        from pipelines.mcmc import mcmc_emu
        sys.exit(mcmc_emu(args))
