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
    def choose_one(type, verbose=False):
        """
        Main function to get the correct Emulator.
        Arguments:
        - type (str): type of emulator;
        - verbose (bool, default: False): verbosity.
        Return:
        - Emulator (object): based on params, get the correct
          emulator and initialize it.
        """
        if type == 'ffnn_emu':
            from src.emu_like.ffnn_emu import FFNNEmu
            return FFNNEmu(verbose=verbose)
        else:
            raise ValueError('Emulator not recognized!')

    def build(self):
        """
        Build the emulator architecture from a
        dictionary of parameters. The parameters
        depend on the architecture.
        """
        return

    def load(self):
        """
        Placeholder for load.
        TODO: write description
        """
        return

    def save(self):
        """
        Placeholder for save.
        TODO: write description
        """
        return

    def train(self):
        """
        Placeholder for train.
        TODO: write description
        """
        return

    def eval(self):
        """
        Placeholder for eval.
        TODO: write description
        """
        return
