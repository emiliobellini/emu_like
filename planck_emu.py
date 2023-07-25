import sys
from tools.io import argument_parser

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = argument_parser()

    # Redirect the run to the correct module
    if args.mode == 'train':
        from tools.train_emu import train_emu
        sys.exit(train_emu(args))
    if args.mode == 'test':
        from tools.test_emu import test_emu
        sys.exit(test_emu(args))
