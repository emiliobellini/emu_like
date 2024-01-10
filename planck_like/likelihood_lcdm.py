"""
This is the base class for the PBJ likelihoods.
It is an example of a working gaussian likelihood,
with two uncorrelated parameters one from the
theory module, and one from the likelihood.
On top of that it contains all the methods with
instructions that should be used to define a new
likelihood.
In particular the '_get_chi2' method is designed
to be general enough so that it has not to be
changed.
"""
import numpy as np
from cobaya.likelihood import Likelihood
import src.defaults as de
import src.io as io
from src.emu import Emulator
from src.scalers import Scaler


class LikelihoodPlanckLcdm(Likelihood):

    def initialize(self):
        """
        Prepare any computation, importing any
        necessary code, files, etc.
        """

        # Init base class
        super().initialize()

        # Load params emulator
        emu_params = io.YamlFile(
            de.file_names['params']['name'],
            root=self.emulator['path'],
            should_exist=True)
        emu_params.read()

        # Define emulator folder
        emu_folder = io.Folder(path=self.emulator['path'])

        # Load scalers
        scalers = emu_folder.subfolder(de.file_names['x_scaler']['folder'])
        scaler_x_path = io.File(de.file_names['x_scaler']['name'],
                                root=scalers)
        scaler_y_path = io.File(de.file_names['y_scaler']['name'],
                                root=scalers)
        self.scaler_x = Scaler.load(scaler_x_path, verbose=self.verbose)
        self.scaler_y = Scaler.load(scaler_y_path, verbose=self.verbose)

        # Call the right emulator
        self.emulator['emu'] = Emulator.choose_one(
            emu_params, emu_folder, verbose=self.verbose)

        # Load emulator
        self.emulator['emu'].load(
            model_to_load=self.emulator['epoch'],
            verbose=self.verbose)

        # Load sample details
        sample_path = emu_folder.subfolder(
            de.file_names['sample_details']['folder'])
        sample_details = io.YamlFile(
            de.file_names['sample_details']['name'],
            root=sample_path)
        sample_details.read()
        self.x_names = sample_details['x_names']


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
