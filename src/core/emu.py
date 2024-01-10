class Emulator(object):
    """
    General emulator class
    """

    def __init__(self, params, output):
        """
        Initialize common sampler features.
        """
        self.name = params['emulator_type']
        self.params = params
        self.output = output
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
            from src.core.ffnn_emu import FFNNEmu
            return FFNNEmu(params, output, verbose=verbose)
        else:
            raise ValueError('Emulator not recognized!')

    def train(self, resume=None, verbose=False, show_plots=False):
        """
        Placeholder for train.
        """
        return
