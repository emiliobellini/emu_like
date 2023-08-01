import argparse
import numpy as np
import os
import re
import yaml
import tools.defaults as de
import tools.printing_scripts as scp


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser('Planck emulator.')

    # Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Options are: '
        '(i) sample: generate the sample for the emulator. '
        '(ii) train: train the emulator. '
        '(iii) test: test the emulator.')
    sample_parser = subparsers.add_parser('sample')
    train_parser = subparsers.add_parser('train')
    test_parser = subparsers.add_parser('test')

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
        '--get_plots', '-p',
        help='Generate diagnostic plots and save them (default: False)',
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
        '--get_plots', '-p',
        help='Generate diagnostic plots and save them (default: False)',
        action='store_true')
    train_parser.add_argument(
        '--resume', '-r',
        type=int,
        help='Resume from a previous run. Number of additional epochs (int)')

    # Test arguments
    test_parser.add_argument('emu_folder', type=str, help='Emulator folder')

    return parser.parse_args()


# ------------------- Folder -------------------------------------------------#

class Folder(object):
    """
    Generic class for folders.

    Arguments:
        - path (str): path to the folder
        - should_exist (bool, optional): check that the folder exists
    """

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
                scp.print_level(1, 'Created folder {}'.format(self.path))
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


# ------------------- Files --------------------------------------------------#

class File(object):
    """
    Generic class for files.

    Arguments:
        - path (str): path to the file
        - should_exist (bool, optional): check that the file exists
    """

    def __init__(self, path, root=None, should_exist=False):
        if root:
            if isinstance(root, Folder):
                root = root.path
            path = os.path.join(root, path)
        self.path = os.path.abspath(path)
        self.parent_folder, self.fname = os.path.split(self.path)
        _, self.ext = os.path.splitext(self.fname)
        # Check existence
        self.exists = os.path.isfile(self.path)
        if should_exist:
            self._exists_or_error()
        # Placeholder for content of the file
        self.content = None
        return

    def _exists_or_error(self):
        """
        Check if a file exists and it is a proper file,
        otherwise raise an error.
        """
        if not os.path.isfile(self.path):
            raise IOError('File {} does not exist!'.format(self.path))
        return

    def create(self, verbose=False):
        """
        Check if a file exists, otherwise create an empty file.
        """
        if not self.exists:
            parent = Folder(path=self.parent_folder)
            parent.create(verbose=verbose)
            f = open(self.path, 'w')
            f.close()
            self.exists = os.path.isfile(self.path)
            if verbose:
                scp.print_level(1, 'Created file {}'.format(self.path))
        self._exists_or_error()
        return self

    def save_array(self, verbose=False):
        np.savetxt(self.path, self.content)
        if verbose:
            scp.print_level(1, 'Created file {}'.format(self.path))
        return


class YamlFile(File):
    """
    Class for yaml files.

    Arguments:
        - path (str): path to the file
        - should_exist (bool, optional): check that the file exists
    """

    def __init__(self, path, root=None, should_exist=False):
        File.__init__(self, path, root, should_exist)
        if self.ext not in ['.yaml', '.yml']:
            raise IOError('{} does not seem to be an YAML file'
                          ''.format(self.path))
        self.content = {}
        return

    def __setitem__(self, item, value):
        self.content[item] = value

    def __getitem__(self, item):
        return self.content[item]

    def keys(self):
        return self.content.keys()

    def read(self):
        """
        Read .yaml file and store its content.
        """
        with open(self.path) as file:
            self.content = yaml.safe_load(file)
        return

    def read_param_or_default(self, param, verbose=False):
        """
        Read param and return its value.
        If not present try to get the default value.
        """
        if not self.content:
            self.read()
        try:
            return self[param]
        except KeyError:
            value = de.default_parameters[param]
            self[param] = value
            if verbose:
                scp.print_level(
                    1, 'Using default value ({}) for {}'.format(value, param))
            return value

    def copy_to(self, name, root=None, header=None, verbose=False):
        """
        Copy YAML file to dest, with the
        header if specified.

        Arguments:
            - dest (str): destination path (relative to root if specified)
            - root (str, optional): root for destination file
            - header (str, optional): strin to be prepended to destination file
        """
        if root:
            dest = os.path.join(root, name)
        params_dest = YamlFile(dest)
        parent = Folder(path=params_dest.parent_folder)
        parent.create(verbose=verbose)
        if header:
            with open(params_dest.path, 'w') as file:
                file.write(header)
        with open(params_dest.path, 'a') as file:
            yaml.dump(self.content, file)
        if verbose:
            scp.print_level(1, 'Created file {}'.format(params_dest.path))
        return

    def check_with(self, ref, to_check, verbose=False):

        # Exception message
        msg = 'Incompatible parameter files! {} is different'

        # Check keys
        for key in to_check.keys():
            if to_check[key]:
                for subk in to_check[key]:
                    if self[key][subk] != ref[key][subk]:
                        raise Exception(msg.format('/'.join([key, subk])))
            else:
                if self[key] != ref[key]:
                    raise Exception(msg.format(key, subk))

        if verbose:
            scp.info('Old parameter file is consistent with the new one')
        return
