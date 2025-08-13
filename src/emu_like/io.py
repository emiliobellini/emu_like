"""
.. module:: io

:Synopsis: Input/output related functions and classes.
:Author: Emilio Bellini

"""

import argparse
import numpy as np
import os
import re
import sys
import yaml
from astropy.io import fits
from collections import OrderedDict


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
        '(iii) test: test the emulator.'
        '(iv) export spectra emulators')
    sample_parser = subparsers.add_parser('sample')
    train_parser = subparsers.add_parser('train')
    mcmc_parser = subparsers.add_parser('mcmc')
    export_parser = subparsers.add_parser('export')

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

    # MCMC arguments
    mcmc_parser.add_argument(
        'params_file',
        type=str,
        help='Parameters file (.yaml)')
    mcmc_parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')

    # Sample arguments
    export_parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input folder')
    export_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output folder')
    export_parser.add_argument(
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

    def list_subfolders(self, patterns=None, unique=False):
        """
        List all subfolders matching any of the patterns (if specified).

        Arguments:
            - patterns (str or list of str, optional): regex patterns
                (subfolder included if any of them is satisfied)
            - unique (bool, optional): check if there is more than
                one subfolder satisfying the patterns

        Return:
            - list of files satisfying the pattern
        """
        if not self.exists:
            filtered = []
        else:
            # List all files in path
            for root, dirs, _ in os.walk(self.path):
                if root == self.path:
                    all = [os.path.join(root, x) for x in dirs]
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
                raise Exception('No subfolders matching patterns')
            elif len(filtered) > 1:
                raise Exception('Multiple subfolders matching patterns')
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

    def subfolder(self, subpath, should_exist=False):
        """
        Define subfolder.

        Returns:
            - Folder class for the resulting path
        """
        path = os.path.join(self.path, subpath)
        return Folder(path=path, should_exist=should_exist)

    def join(self, subpath):
        """
        Join folder with subpath.

        Returns:
            - String with location of the resulting path
        """
        path = os.path.join(self.path, subpath)
        return path


# ------------------- Fits Files ---------------------------------------------#

class FitsFile(object):
    """
    Class for fits files.
    """

    def __init__(self, fname=None, root=None):
        # Define path
        if root is None:
            self.path = fname
        else:
            self.path = os.path.abspath(os.path.join(root, fname))
        # Check existence
        self.exists = os.path.isfile(self.path)
        # Check is fits
        is_fits = self.path.endswith('.fits') or self.path.endswith('.fits.gz')
        if not is_fits:
            raise ValueError('Expected .fits file, found {}'.format(self.path))
        return

    def _flatten_dict(self, nested_dict, delimiter='__'):
        split_dict = {}
        flat_dict = self._flatten_dict_recursive(nested_dict, delimiter=delimiter)
        # In astropy, keys can not be longer than 8 characters. We then create a flat
        # dict where both keys and values are values. This dictionary will have keys
        # starting with two delimiters for the keys of the previous step dictionary,
        # and keys starting with one delimiter for the values of the previous dictionary.
        # To fix the correspondence each key ends with a different integer,
        for nkey, (key, val) in enumerate(flat_dict.items()):
            split_dict['{}{}{}'.format(delimiter, delimiter, nkey)] = key
            split_dict['{}{}'.format(delimiter, nkey)] = val
        return split_dict

    def _flatten_dict_recursive(self, nested_dict, parent_key='', delimiter='__'):
        """Flatten a nested dictionary, preserving key order."""
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{delimiter}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict_recursive(value, new_key, delimiter).items())
            else:
                items.append((new_key, value))
        return OrderedDict(items)

    def _unflatten_dict(self, flat_dict, delimiter='__'):
        current_dict = {}
        # We first fix the correspondence between keys and values to get a list of
        # flattened keys and values (si discussion in _flatten_dict above).
        for key_flat in flat_dict.keys():
            if key_flat.startswith('{}{}'.format(delimiter, delimiter)):
                key = flat_dict[key_flat]
                val = flat_dict[key_flat[len(delimiter):]]
                current_dict[key] = val
        """Reconstruct a nested dictionary from flattened keys, preserving order."""
        result = OrderedDict()
        for key, value in current_dict.items():
            parts = key.split(delimiter)
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = OrderedDict()
                d = d[part]
            d[parts[-1]] = value
        return result

    def _delistify(self, flat_dict, delimiter='_*_'):
        # fits header do not accept lists as values.
        # here we convert them into strings, assuming
        # ndim <= 2.
        delimiter2 = 2*delimiter
        for key, val1 in flat_dict.items():
            if isinstance(val1, list):
                if all([isinstance(x, str) for x in val1]):
                    val1 = delimiter2.join(val1)
                    flat_dict[key] = val1
                if all([isinstance(x, list) for x in val1]):
                    tmp = []
                    for val2 in val1:
                        tmp.append(delimiter.join([str(x) for x in val2]))
                    flat_dict[key] = delimiter2.join(tmp)
        return flat_dict

    def _listify(self, flat_dict, delimiter='_*_'):
        # fits header do not accept lists as values.
        # here we convert back delistified lists,
        # assuming ndim <= 2.
        current_dict = {}
        delimiter2 = 2*delimiter
        for key, val1 in flat_dict.items():
            if isinstance(val1, str) and delimiter2 in val1:
                current_dict[key] = val1.split(delimiter2)
            else:
                current_dict[key] = val1
        for key, val1 in current_dict.items():
            if isinstance(val1, str) and delimiter in val1:
                current_dict[key] = val1.split(delimiter)
            elif isinstance(val1, list):
                for nval2, val2 in enumerate(val1):
                    if isinstance(val2, str) and delimiter in val2:
                        current_dict[key][nval2] = val2.split(delimiter)
                    else:
                        current_dict[key][nval2] = val2
        for key, val1 in current_dict.items():
            if isinstance(val1, str):
                current_dict[key] = self._floatify(val1)
            elif isinstance(val1, list):
                for nval2, val2 in enumerate(val1):
                    if isinstance(val2, str):
                        current_dict[key][nval2] = self._floatify(val2)
                    elif isinstance(val2, list):
                        current_dict[key][nval2] = [self._floatify(x) for x in val2]
                    else:
                        current_dict[key][nval2] = val2
        return current_dict

    def _floatify(self, val_string):
        try:
            val = float(val_string)
        except ValueError:
            val = val_string
        return val

    def write(self, data, header, name, verbose=False):
        # Create parent folder
        try:
            Folder(os.path.dirname(self.path)).create()
        except FileExistsError:
            pass
        # We assume that header is either a dictionary, or a fits.Header
        if isinstance(header, dict):
            header = self._flatten_dict(header)
            header = self._delistify(header)
            header = fits.Header(header)
        # Create first HDU
        if not self.exists:
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            hdul.writeto(self.path)
            self.exists = True
        else:
            # Open the file and append
            with fits.open(self.path, mode='append') as hdul:
                hdul.append(fits.ImageHDU(data, name=name, header=header))
        if verbose:
            print_level(1, 'Appended {} to {}'.format(name.upper(), os.path.relpath(self.path)))
            sys.stdout.flush()
        return

    def append(self, data, name, header=None):
        if isinstance(header, dict):
            header = self._flatten_dict(header)
            header = self._delistify(header)
            header = fits.Header(header)
        # If array already exists, append data to it,
        # otherwise create array.
        try:
            with fits.open(self.path, mode='update') as hdul:
                hdul[name].data = np.vstack([hdul[name].data, data])
        except KeyError:
            with fits.open(self.path, mode='append') as hdul:
                hdul.append(fits.ImageHDU(data, name=name, header=header))
        return

    def print_info(self):
        """ Print on screen fits file info.

        Args:
            fname: path of the input file.

        Returns:
            None

        """

        with fits.open(self.path) as hdul:
            print(hdul.info())
            sys.stdout.flush()
        return

    def get_header(self, name, unflat_dict=True):
        """ Open a fits file and return the header from name.

        Args:
            name: name of the data we want to extract.

        Returns:
            header.

        """
        with fits.open(self.path) as fn:
            if unflat_dict:
                hd = self._listify(fn[name].header)
                hd = self._unflatten_dict(hd)
            else:
                hd = fn[name].header
        return hd

    def get_data(self, name):
        """ Open a fits file and return the header fromname.

        Args:
            name: name of the data we want to extract.

        Returns:
            header.

        """
        with fits.open(self.path) as fn:
            return fn[name].data


# ------------------- Yaml Files ---------------------------------------------#

class YamlFile(object):
    """
    Class for yaml files.
    """

    def __init__(self, fname=None, root=None):
        # Define path
        if root is None:
            self.path = fname
        else:
            self.path = os.path.abspath(os.path.join(root, fname))
        # Check existence
        self.exists = os.path.isfile(self.path)
        # Check is yaml
        is_yaml = self.path.endswith('.yaml')
        if not is_yaml:
            raise Exception('Expected .yaml file, found {}'.format(self.path))
        # Defaults
        self.default_name = 'params.yaml'
        self.default_header = (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n')
        return

    def __setitem__(self, item, value):
        self.content[item] = value

    def __getitem__(self, item):
        return self.content[item]

    def __repr__(self):
        return self.content

    def __str__(self):
        return str(self.content)

    def keys(self):
        return self.content.keys()

    def read(self, fname=None, root=None):
        """
        Read .yaml file and store it into a dictionary.
        Arguments:
        - fname (str): path to the parameters file;
        - root (str, default: None): root of the file;
          blocks with default structure.
        """
        # Define path
        if fname is not None:
            if root is None:
                self.path = fname
            else:
                self.path = os.path.abspath(os.path.join(root, fname))
        # Check existence
        self.exists = os.path.isfile(self.path)

        if not self.exists:
            raise FileNotFoundError(
                'The file you want to read ({}) does not exists!'.format(self.path))

        with open(self.path) as file:
            self.content = yaml.safe_load(file)
        
        return self

    def write(self, fname=None, root=None, header=None, overwrite=False, verbose=False):
        """
        Save parameter to path, with the header if specified.
        Arguments:
        - fname (str): destination file name, relative to root if specified;
        - root (str, default: None): root where to save the file;
        - header (str, optional): string to be prepended to destination file;
        - overwrite (bool, default: False): overwrite already existing file;
        - verbose (bool, default: False): verbosity.
        """

        # Define path
        if root is None:
            self.path = fname
        else:
            self.path = os.path.abspath(os.path.join(root, fname))
        # Check existence
        self.exists = os.path.isfile(self.path)

        if self.exists and not overwrite:
            raise FileNotFoundError(
                'The file you want to read ({}) already exists!'.format(self.path))

        if header is None:
            header = self.default_header
        # Create root folder and join path
        try:
            Folder(os.path.dirname(self.path)).create()
        except FileExistsError:
            pass

        if header:
            with open(self.path, 'w') as file:
                file.write(header)
        with open(self.path, 'a') as file:
            yaml.safe_dump(self.content, file, sort_keys=False)
        if verbose:
            print_level(1, 'Saved parameters at: {}'.format(self.path))
        return



# ------------------- Scripts ------------------------------------------------#

def write_red(msg):
    return '\033[1;31m{}\033[00m'.format(msg)


def write_green(msg):
    return '\033[1;32m{}\033[00m'.format(msg)


def warning(msg):
    prepend = write_red('[WARNING]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def info(msg):
    prepend = write_green('[info]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def print_level(num, msg, arrow=True):
    if num > 0:
        if arrow:
            prepend = write_green(num*'----' + '> ')
        else:
            prepend = (4*num+2)*' '
    else:
        prepend = ''
    print('{}{}'.format(prepend, msg), flush=True)
    return
