"""
.. module:: emu

:Synopsis: Input/output related functions and classes.
:Author: Emilio Bellini

"""

import argparse
import os
import re
import sys
from astropy.io import fits


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

class Path(object):
    """
    Generic class for paths.
    """
    def __init__(self, path=None, exists=False, check_is_folder=False):
        # Store location
        if path:
            self.path = os.path.abspath(path)
            self.parent_folder, self.fname = os.path.split(self.path)
            self.name, self.ext = os.path.splitext(self.fname)
            if self.ext:
                self.isfile = True
                self.isfolder = False
            else:
                self.isfile = False
                self.isfolder = True
        else:
            self.path = None
            self.parent_folder = None
            self.fname = None
            self.name = None
            self.ext = None
            self.isfile = None
            self.isfolder = None
        # Check existence
        self.exists = os.path.exists(self.path)
        if exists:
            if path:
                self.exists_or_error()
            else:
                raise ValueError('Can not check existence of Path. Argument '
                                 'exists=True but path to the file '
                                 'not provided!')
        if check_is_folder:
            if not self.isfolder:
                raise ValueError('This should be a folder but it is not!')
        return

    def exists_or_error(self):
        """
        Check if a path exists, otherwise raise an error.
        """
        assert self.exists, 'File {} does not exist!'.format(self.path)
        return

    def exists_or_create(self):
        """
        Check if a path exists, otherwise create it.
        If the path is a file creates an empty file,
        if it is a folder create the folder.
        """
        if not os.path.exists(self.path):
            if self.isfile:
                if not os.path.exists(self.parent_folder):
                    parent = Path(path=self.parent_folder)
                    parent.exists_or_create()
                f = open(self.path, 'w')
                f.close()
            elif self.isfolder:
                os.makedirs(self.path)
        return


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


# ------------------- Files --------------------------------------------------#

class File(Path):
    """
    Generic class for files.
    """

    def __init__(self, path=None, exists=False):
        Path.__init__(self, path, exists)
        # Check existence
        self.exists = os.path.isfile(path)
        # Placeholder for content
        self.content = None
        return

    def head(self, lines=1):
        """
        Imitates the bash head command
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            f.seek(0, 2)     # go to end of file
            total_bytes = f.tell()
            lines_found, total_bytes_scanned = 0, 0
            while (lines+1 > lines_found and
                    total_bytes > total_bytes_scanned):
                byte_block = min(1024, total_bytes-total_bytes_scanned)
                f.seek(total_bytes_scanned, 0)
                total_bytes_scanned += byte_block
                lines_found += f.read(byte_block).count('\n')
            f.seek(0, 0)
            line_list = list(f.readlines(total_bytes_scanned))
            line_list = [x.rstrip() for x in line_list[:lines]]
        return line_list

    def tail(self, lines=1):
        """
        Imitates the bash tail command
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            f.seek(0, 2)     # go to end of file
            total_bytes = f.tell()
            lines_found, total_bytes_scanned = 0, 0
            while (lines+1 > lines_found and
                    total_bytes > total_bytes_scanned):
                byte_block = min(1024, total_bytes-total_bytes_scanned)
                f.seek(total_bytes-total_bytes_scanned-byte_block, 0)
                lines_found += f.read(byte_block).count('\n')
                total_bytes_scanned += byte_block
            f.seek(total_bytes-total_bytes_scanned, 0)
            line_list = list(f.readlines())
            line_list = [x.rstrip() for x in line_list[-lines:]]
        return line_list

    def read(self, size=-1):
        """
        Read the file.
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            content = f.read(size)
        self.content = content
        return content

    def readlines(self, size=-1):
        """
        Read each line of the file.
        """
        with open(self.path, 'r') as f:
            lines = f.readlines(size)
        return lines

    def read_header(self, comments='#'):
        """
        Read the header of a file, i.e. from the beginning of the file
        all lines that start with the comments symbol.
        """
        # Check
        self.exists_or_error()
        # Comment string
        if type(comments) is str:
            comments = [comments]
        assert type(comments) is list, 'Wrong format for comments '\
            'in File.read_header(). It should be a list, got {}!'\
            ''.format(type(comments))
        # Main body
        is_head = True
        head = []
        f = open(self.path, 'r')
        ln = f.readline()
        while ln and is_head:
            if any([re.match('{}.*'.format(x), ln) for x in comments]):
                head.append(ln)
            else:
                is_head = False
            ln = f.readline()
        f.close()
        head = ''.join(head)
        return head

    def write(self, content=None, path=None, overwrite=False):
        """
        Write the string 'content' into the file.
        """
        if not path:
            path = self.path
        if not content:
            content = self.content
        # Check
        assert path, 'To write a file you should specify a path'
        assert content, 'No content to write'
        if not overwrite and self.exists:
            raise IOError('File {} exists, if you really want to overwrite '
                          'it, use overwrite argument.'.format(path))
        # Main body
        with open(path, 'w') as f:
            f.write(content)
        print('--> File saved at {}!'.format(path))
        return

    def append(self, content=None, path=None):
        """
        Append the string 'content' into the file.
        """
        if not path:
            path = self.path
        if not content:
            content = self.content
        # Check
        assert path, 'To write a file you should specify a path'
        assert content, 'No content to write'
        if not self.exists:
            raise IOError('Can not append to file {}, since it does not '
                          'exists. Use write if you want to create it.'
                          ''.format(path))
        # Main body
        with open(path, 'a') as f:
            f.write(content)
        print('--> Content appended to file at {}!'.format(path))
        return

    def remove(self):
        """
        Remove file and reinitialize the class
        """
        os.remove(self.path)
        self.__init__(self.path)
        return


# ------------------- Fits Files ---------------------------------------------#

class FitsFile(File):
    """
    Class for fits files.
    """

    def __init__(self, path=None, exists=False):
        File.__init__(self, path, exists)
        # Placeholder for content
        self.content = {}
        # Check is fits
        test1 = self.ext == '.fits'
        test2 = self.ext == '.gz' and os.path.splitext(self.name)[1] == '.fits'
        assert test1 or test2, 'Expected .fits file, found {}'.format(
            self.ext)
        return

    def read(self, dtype=None):
        """ Open a fits file and read all data from it.

        Args:
            key: name of the data we want to read.

        Returns:
            array with data for name.

        """
        with fits.open(self.path) as fn:
            for key in fn:
                if dtype:
                    self.content[key] = fn[key].data.astype(dtype)
                else:
                    self.content[key] = fn[key].data
        return

    def read_key(self, key, dtype=None):
        """
        Open a fits file and read a key data from it.

        Args:
            key: name of the data we want to read.
        """
        with fits.open(self.path) as fn:
            if dtype:
                return fn[key].data.astype(dtype)
            else:
                return fn[key].data

    def get_keys(self):
        """ Get keys from fits file.

        Args:
            fname: path of the data file.

        Returns:
            list of keys.

        """
        with fits.open(self.path) as fn:
            return [x.name for x in fn]

    def get_header(self, name):
        """ Open a fits file and read header from it.

        Args:
            fname: path of the data file.
            name: name of the data we want to extract.

        Returns:
            header.

        """
        with fits.open(self.path) as fn:
            return fn[name].header

    def write(self, array, name, type='image', header=None, verbose=False):
        """ Write an array to a fits file.

        Args:
            fname: path of the input file.
            array: array to save.
            name: name of the image.

        Returns:
            None

        """

        warning = False

        # If file does not exist, create it
        if not os.path.exists(self.path):
            hdul = fits.HDUList([fits.PrimaryHDU()])
            hdul.writeto(self.path)
        # Open the file
        with fits.open(self.path, mode='update') as hdul:
            try:
                hdul.__delitem__(name)
            except KeyError:
                pass
            if type == 'image':
                hdul.append(fits.ImageHDU(array, name=name, header=header))
            elif type == 'table':
                hdul.append(array)
            else:
                print('Type '+type+' not recognized! Data not saved to file!')
                return True
        if verbose:
            print('Appended ' + name.upper() + ' to ' + os.path.relpath(self.path))
            sys.stdout.flush()
        return warning

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
