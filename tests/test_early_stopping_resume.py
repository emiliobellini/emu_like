"""Early-stopping decisions must survive repeated training restarts."""
import unittest
from unittest.mock import Mock

import numpy as np
from tensorflow import keras

from emu_like.ffnn_emu import (
    EarlyStoppingHistory, FFNNEmu, RelativeEarlyStopping,
)
from emu_like.sobolev_ffnn_emu import SobolevFFNNEmu


def model():
    result = keras.Sequential([
        keras.Input(shape=(1,)), keras.layers.Dense(1, use_bias=False),
    ])
    result.set_weights([np.array([[7.]], dtype=np.float32)])
    result.stop_training = False
    return result


def stopping(relative, patience=4):
    if relative:
        return RelativeEarlyStopping(patience=patience)
    return keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=patience,
        restore_best_weights=True)


def run(losses, relative, restart_epochs=()):
    net = model()
    callback = stopping(relative)
    callback.set_model(net)
    callback.on_train_begin()
    for epoch, loss in enumerate(losses):
        if epoch in restart_epochs:
            callback = stopping(relative)
            recovery = EarlyStoppingHistory(
                callback, list(range(epoch)), losses[:epoch])
            if recovery.exhausted:
                return epoch - 1
            callback.set_model(net)
            recovery.set_model(net)
            callback.on_train_begin()
            recovery.on_train_begin()
        callback.on_epoch_end(epoch, {'val_loss': loss})
        if net.stop_training:
            return epoch
    return None


class EarlyStoppingResumeTests(unittest.TestCase):
    def test_repeated_restarts_match_uninterrupted(self):
        for relative in (False, True):
            for losses in ([1., .9, .91, .92, .93, .94, .95],
                           [1., .9996, .9992, .9988, .9987, .9986,
                            .9985, .9984, .9983, .9982],
                           [1., .9, .91, .92, .8, .81, .82, .83, .84]):
                with self.subTest(relative=relative, losses=losses):
                    self.assertEqual(
                        run(losses, relative),
                        run(losses, relative, restart_epochs=(2, 4, 6, 8)))

    def test_already_exhausted_and_single_epoch(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                recovery = EarlyStoppingHistory(
                    stopping(relative), list(range(5)), [1.] * 5)
                self.assertTrue(recovery.exhausted)
                recovery = EarlyStoppingHistory(stopping(relative), [0], [1.])
                self.assertFalse(recovery.exhausted)
                self.assertEqual(recovery.state['wait'], 0)

    def test_restores_loaded_weights_without_new_best(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                net = model()
                callback = stopping(relative)
                recovery = EarlyStoppingHistory(callback, [0, 1], [1., 1.1])
                callback.set_model(net)
                recovery.set_model(net)
                callback.on_train_begin()
                recovery.on_train_begin()
                net.set_weights([np.array([[99.]], dtype=np.float32)])
                for epoch in range(2, 5):
                    callback.on_epoch_end(epoch, {'val_loss': 1.1})
                callback.on_train_end()
                np.testing.assert_array_equal(net.get_weights()[0], [[7.]])

    def test_both_emulators_install_recovery(self):
        for emulator_type in (FFNNEmu, SobolevFFNNEmu):
            for relative in (False, True):
                with self.subTest(emulator=emulator_type, relative=relative):
                    emulator = emulator_type()
                    emulator.epochs = list(range(5))
                    emulator.val_loss = [1.] * 5
                    callbacks = emulator._callbacks(
                        patience=4, relative_improvement=relative,
                        reduce_learning_rate=False)
                    self.assertTrue(emulator._early_stopping_exhausted(callbacks))
                    callbacks = emulator._callbacks(
                        patience=None, reduce_learning_rate=False)
                    self.assertFalse(emulator._early_stopping_exhausted(callbacks))

    def test_exhausted_training_does_not_fit_or_save(self):
        emulator = FFNNEmu()
        emulator.epochs = list(range(5))
        emulator.val_loss = [1.] * 5
        emulator.model = Mock()
        emulator.save = Mock()
        emulator.train(Mock(), epochs=100, learning_rate=.01,
                       patience=4, reduce_learning_rate=False)
        emulator.model.fit.assert_not_called()
        emulator.save.assert_not_called()

    def test_new_improvement_resets_recovered_wait(self):
        for relative in (False, True):
            callback = stopping(relative)
            recovery = EarlyStoppingHistory(callback, [0, 1, 2], [1., 1.1, 1.2])
            net = model()
            callback.set_model(net)
            recovery.set_model(net)
            callback.on_train_begin()
            recovery.on_train_begin()
            self.assertEqual(callback.wait, 2)
            callback.on_epoch_end(3, {'val_loss': .5})
            self.assertEqual(callback.wait, 0)
            self.assertFalse(net.stop_training)


if __name__ == '__main__':
    unittest.main()
