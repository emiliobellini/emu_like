"""
.. module:: emu

:Synopsis: Input/output related functions and classes.
:Author: Emilio Bellini

"""

import argparse
import os
import re


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser('Likelihood emulator.')

    # Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Options are: '
        '(i) sample: generate the sample for the emulator. '
        '(ii) train: train the emulator. '
        '(iii) test: test the emulator.')
    sample_parser = subparsers.add_parser('sample')
    train_parser = subparsers.add_parser('train')
    mcmc_parser = subparsers.add_parser('mcmc')

    # Sample arguments
    sample_parser.add_argument(
        'params_file',
        type=str,
        help='Parameters file (.yaml)')
    sample_parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')
    sample_parser.add_argument(
        '--resume', '-r',
        help='Resume from a previous run.',
        action='store_true')

    # Train arguments
    train_parser.add_argument(
        'params_file',
        type=str,
        help='Parameters file (.yaml)')
    train_parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')
    train_parser.add_argument(
        '--resume', '-r',
        help='Resume from a previous run.',
        action='store_true')
    train_parser.add_argument(
        '--additional_epochs', '-e',
        type=int,
        default=0,
        help='Number of additional epochs (int)')
    train_parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=1.e-3,
        help='New learning rate (float)')

    # Test arguments
    mcmc_parser.add_argument(
        'params_file',
        type=str,
        help='Parameters file (.yaml)')
    mcmc_parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')

    return parser.parse_args()


# ------------------- Folder -------------------------------------------------#

class Folder(object):
    """
    Generic class for folders.
    Here we implemented some ad-hoc method
    to ease some common task with folders.

    Arguments:
        - path (str): path to the folder
        - should_exist (bool, optional): check that the folder exists
    """

    def __repr__(self):
        return self.path

    def __str__(self):
        return str(self.path)

    def __init__(self, path, should_exist=False):
        self.path = os.path.abspath(path)
        # Check existence
        self.exists = os.path.isdir(self.path)
        if should_exist:
            self._exists_or_error()

    def _exists_or_error(self):
        """
        Check if a folder exists and it is a proper
        directory, otherwise raise an error.
        """
        if not os.path.isdir(self.path):
            raise IOError('Folder {} does not exist!'.format(self.path))
        return

    def create(self, verbose=False):
        """
        Check if a folder exists, otherwise create it.

        Returns:
            - self: the same object
        """
        if not self.exists:
            os.makedirs(self.path)
            self.exists = os.path.isdir(self.path)
            if verbose:
                print_level(1, 'Created folder {}'.format(self.path))
        self._exists_or_error()
        return self

    def list_files(self, patterns=None, unique=False):
        """
        List all files matching any of the patterns (if specified).

        Arguments:
            - patterns (str or list of str, optional): regex patterns
                (file included if any of them is satisfied)
            - unique (bool, optional): check if there is more than
                one file satisfying the patterns

        Return:
            - list of files satisfying the pattern
        """
        if not self.exists:
            filtered = []
        else:
            # List all files in path
            for root, _, files in os.walk(self.path):
                if root == self.path:
                    all = [os.path.join(root, x) for x in files]
            # Filter with pattern
            if patterns:
                if isinstance(patterns, str):
                    patterns = [patterns]
                filtered = []
                for pattern in patterns:
                    filtered += [x for x in all if re.match(pattern, x)]
            else:
                filtered = all
        # Check uniqueness
        if unique:
            if len(filtered) == 0:
                raise Exception('No files matching patterns')
            elif len(filtered) > 1:
                raise Exception('Multiple files matching patterns')
        return filtered

    def is_empty(self):
        """
        Check if a folder is empty or not.

        Return:
            - True if folder is empty (or it does not exist), False otherwise
        """
        if not self.exists:
            return True
        if self.list_files():
            return False
        else:
            return True

    def subfolder(self, subpath):
        """
        Define subfolder.

        Returns:
            - Folder class for the resulting path
        """
        path = os.path.join(self.path, subpath)
        return Folder(path=path)

    def join(self, subpath):
        """
        Join folder with subpath.

        Returns:
            - String with location of the resulting path
        """
        path = os.path.join(self.path, subpath)
        return path


# ------------------- Scripts ------------------------------------------------#

def write_red(msg):
    return '\033[1;31m{}\033[00m'.format(msg)


def write_green(msg):
    return '\033[1;32m{}\033[00m'.format(msg)


def warning(msg):
    prepend = write_red('[WARNING]')
    print('{} {}'.format(prepend, msg))
    return


def info(msg):
    prepend = write_green('[info]')
    print('{} {}'.format(prepend, msg))
    return


def print_level(num, msg, arrow=True):
    if num > 0:
        if arrow:
            prepend = write_green(num*'----' + '> ')
        else:
            prepend = (4*num+2)*' '
    else:
        prepend = ''
    print('{}{}'.format(prepend, msg))
    return
