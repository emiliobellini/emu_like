import yaml


class Params(object):
    """
    Class dealing with parameters.
    """

    def __init__(self):
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
