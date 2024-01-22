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

    def keys(self):
        return self.content.keys()

    def load(self, path):
        """
        Load .yaml file and store it into a dictionary.
        Arguments:
        - path (str): path to the parameters file.
        """
        with open(path) as file:
            self.content = yaml.safe_load(file)
        return self
    
    def save(self, path, header=None, verbose=False):
        """
        Save parameter to path, with the header if specified.

        Arguments:
            - path (str): destination path
            - header (str, optional): string to be prepended
              to destination file.
        """
        if header:
            with open(path, 'w') as file:
                file.write(header)
        with open(path, 'a') as file:
            yaml.safe_dump(self.content, file, sort_keys=False)
        if verbose:
            io.print_level(1, 'Saved parameters at: {}'.format(path))
        return
