"""Sobolev FFNN emulator.

This is currently an interface placeholder.  It deliberately inherits the
ordinary FFNN behaviour so that the training pipeline can select, construct,
and save the new emulator type before the paired P(k)-growth dataset and
derivative-aware training step are introduced.
"""

from .ffnn_emu import FFNNEmu


class SobolevFFNNEmu(FFNNEmu):
    """Placeholder for an FFNN constrained by redshift derivatives."""

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.name = 'sobolev_ffnn_emu'

