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
        return

    def __setitem__(self, item, value):
        self.content[item] = value

    def __getitem__(self, item):
        return self.content[item]

    def __repr__(self):
        return self.content

    def __str__(self):
        return str(self.content)

    def _fill_missing(self):
        default_dict = {
            'output': None,
            'sampler': {
                'name': None,
                'args': None,
            },
            'train_generator': {
                'name': None,
                'args': None,
                'outputs': None,
            },
            'params': None,
        }
        for key1 in default_dict:
            if key1 not in self.content:
                self.content[key1] = default_dict[key1]
            if isinstance(default_dict[key1], dict):
                for key2 in default_dict[key1]:
                    if key2 not in self.content[key1]:
                        self.content[key1][key2] = default_dict[key1][key2]
        return

    def keys(self):
        return self.content.keys()

    def load(self, path, root=None, fill_missing=False):
        """
        Load .yaml file and store it into a dictionary.
        Arguments:
        - path (str): path to the parameters file;
        - root (str, default: None): root of the file;
        - fill_missing (bool, default: False): fill missing
          blocks with default structure.
        """

        # Join root
        if root:
            path = os.path.join(root, path)

        with open(path) as file:
            self.content = yaml.safe_load(file)
        
        if fill_missing:
            self._fill_missing()
        return self

    def save(self, path, root=None, header=None, verbose=False):
        """
        Save parameter to path, with the header if specified.
        Arguments:
        - path (str): destination file name, relative to root if specified;
        - root (str, default: None): root where to save the file;
        - header (str, optional): string to be prepended to destination file;
        - verbose (bool, default: False): verbosity.
        """

        # Join root
        if root:
            path = os.path.join(root, path)

        if header:
            with open(path, 'w') as file:
                file.write(header)
        with open(path, 'a') as file:
            yaml.safe_dump(self.content, file, sort_keys=False)
        if verbose:
            io.print_level(1, 'Saved parameters at: {}'.format(path))
        return
