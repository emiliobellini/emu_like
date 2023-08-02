import sys
from tools.io import argument_parser

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
    if args.mode == 'test':
        from pipelines.test_mcmc import test_mcmc_emu
        sys.exit(test_mcmc_emu(args))
