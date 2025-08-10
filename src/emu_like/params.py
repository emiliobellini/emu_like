"""
.. module:: params

:Synopsis: Module with the Params class, dealing with
    dictionaries of parameters.
:Author: Emilio Bellini

"""

import os
import yaml
from . import io as io


class Params(object):
    """
    Class dealing with parameters.
    """

    def __init__(self, content=None):
        if content:
            self.content = content
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

    def load(self, fname=None, root=None):
        """
        Load .yaml file and store it into a dictionary.
        Arguments:
        - fname (str): path to the parameters file;
        - root (str, default: None): root of the file;
        - fill_missing (bool, default: False): fill missing
          blocks with default structure.
        """

        # Join root
        if fname is None:
            fname = self.default_name
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)

        with open(path) as file:
            self.content = yaml.safe_load(file)
        
        return self

    def save(self, fname=None, root=None, header=None, verbose=False):
        """
        Save parameter to path, with the header if specified.
        Arguments:
        - fname (str): destination file name, relative to root if specified;
        - root (str, default: None): root where to save the file;
        - header (str, optional): string to be prepended to destination file;
        - verbose (bool, default: False): verbosity.
        """

        if fname is None:
            fname = self.default_name
        if header is None:
            header = self.default_header
        # Create root folder and join path
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)
        io.Folder(os.path.dirname(path)).create()

        if header:
            with open(path, 'w') as file:
                file.write(header)
        with open(path, 'a') as file:
            yaml.safe_dump(self.content, file, sort_keys=False)
        if verbose:
            io.print_level(1, 'Saved parameters at: {}'.format(path))
        return
