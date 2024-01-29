"""
.. module:: likelihood_lcdm

:Synopsis: Module containing the likelihood for Cobaya sampling
:Author: Emilio Bellini
TODO: this does not work

"""

import numpy as np
from cobaya.likelihood import Likelihood
from .. import defaults as de
from .. import io as io
from ..emu import Emulator
from ..params import Params


class LikelihoodPlanckLcdm(Likelihood):

    def initialize(self):
        """
        Prepare any computation, importing any
        necessary code, files, etc.
        """

        # Init base class
        super().initialize()

        # Read params
        emu_params = Params().load(
            de.file_names['params']['name'],
            root=self.emulator['path']
        )

        # Call the right emulator
        self.emu = Emulator.choose_one(emu_params['emulator']['type'],
                                       verbose=self.verbose)
        # Load emulator
        self.emu.load(emu_params['output'], model_to_load='best',
                      verbose=self.verbose)

        # Define emulator folder
        emu_folder = io.Folder(path=self.emulator['path'])

        # Call the right emulator
        self.emulator['emu'] = Emulator.choose_one(
            emu_params, emu_folder, verbose=self.verbose)

        # Load emulator
        self.emulator['emu'].load(
            model_to_load=self.emulator['epoch'],
            verbose=self.verbose)

    def get_requirements(self):
        """
        Return dictionary specifying quantities
        calculated by a theory code that are needed

        e.g. {'Cl': {'tt': 2500}, 'H0': None}
        """
        return {}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance
        parameter values params_values and return
        a log-likelihood.
        """
        params = [params_values[x] for x in self.x_names]

        chi2 = self.evaluate_emulator(
            params,
            self.emulator['emu'].model,
            self.scaler_x,
            self.scaler_y)[0, 0]
        return -chi2/2.

    def evaluate_emulator(self, x, model, scaler_x, scaler_y):
        x_reshaped = np.array([x])
        if scaler_x:
            x_scaled = scaler_x.transform(x_reshaped)
        else:
            x_scaled = x_reshaped
        y_scaled = model(x_scaled, training=False)
        y = scaler_y.inverse_transform(y_scaled)
        return y
