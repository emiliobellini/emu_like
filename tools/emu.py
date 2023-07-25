class Emulator(object):
    """
    Placeholder for emulators
    """

    def __init__(self, params, output):
        """
        Initialize common sampler features.
        """
        self.name = params['emulator_type']
        self.output = output
        self.params = params
        return

    @staticmethod
    def choose_one(params, output, verbose=False):
        """
        Main function to get the correct Emulator.

        Arguments:
            - params (dict): parameters used by the emulator.

        Return:
            - Emulator (object): based on params, get the correct
              emulator and initialize it.
        """
        if params['emulator_type'] == 'ffnn_emu':
            from tools.ffnn_emu import FFNNEmu
            return FFNNEmu(params, output, verbose=verbose)
        else:
            raise ValueError('Emulator not recognized!')

    def train(self, verbose=False, show_plots=False):
        """
        Placeholder for train.
        """
        return

    def test(self, verbose=False, show_plots=False):
        """
        Placeholder for test.
        """
        return

    def load(self, path=None, verbose=False):
        """
        Placeholder for load.
        """
        return

    def save(self, path=None, verbose=False):
        """
        Placeholder for save.
        """
        return
