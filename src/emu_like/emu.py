"""
.. module:: emu

:Synopsis: Module with the base emulator class.
:Author: Emilio Bellini

"""

from . import io as io


class Emulator(object):
    """
    Base emulator class.
    The main goal of this class is to redirect
    to the correct emulator with 'choose_one'
    """

    def __init__(self):
        """
        Nothing here.
        """
        return

    @staticmethod
    def choose_one(emu_type, verbose=False):
        """
        Main function to get the correct Emulator.
        Arguments:
        - type (str): type of emulator;
        - verbose (bool, default: False): verbosity.
        Return:
        - Emulator (object): based on params, get the correct
          emulator and initialize it.
        """
        if emu_type == 'ffnn_emu':
            from .ffnn_emu import FFNNEmu
            return FFNNEmu(verbose=verbose)
        else:
            raise ValueError('Emulator not recognized!')

    def load(self, path, model_to_load='best', verbose=False):
        """
        Load a model.
        This should store the model into the 'model'
        attribute. Along with that, it should load all
        the other elements necessary to run the emulator,
        e.g. x_scaler, y_scaler and and initial_epoch.

        This method redirects to the correct emulator.
        """
        params = io.YamlFile(root=path).load()
        if params['emulator']['type'] == 'ffnn_emu':
            from .ffnn_emu import FFNNEmu
            emu = FFNNEmu(verbose=verbose)
            emu = emu.load(path, model_to_load=model_to_load, verbose=verbose)
            return emu
        else:
            raise ValueError('Emulator not recognized!')

    def save(self):
        """
        Save a model in a folder.
        This method should be compatible with the attributes
        loaded with 'load'.
        """
        return

    def build(self):
        """
        Build the emulator architecture from a
        dictionary of parameters. The parameters
        depend on the architecture.
        """
        return

    def train(self):
        """
        Train the emulator.
        This can be used after calling 'build', to train
        a freshly new created architecture, or after 'load'
        to resume training.
        """
        return

    def eval(self):
        """
        Evaluate the emulator at a single point,
        and return its output.
        """
        return
