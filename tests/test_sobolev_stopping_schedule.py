"""Sobolev convergence decisions use only the final, fixed objective."""
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from tensorflow import keras

from emu_like.ffnn_emu import EarlyStoppingHistory, StrictStateCheckpoint
from emu_like.sobolev_ffnn_emu import SobolevFFNNEmu


def emulator():
    result = SobolevFFNNEmu()
    result.sobolev_params = {'fk_warmup_epochs': 3, 'fk_ramp_epochs': 2}
    return result


def network():
    net = keras.Sequential([
        keras.Input(shape=(1,)), keras.layers.Dense(1, use_bias=False),
    ])
    net.compile(optimizer=keras.optimizers.SGD(learning_rate=1.), loss='mse')
    net.stop_training = False
    return net


def run(relative, restarts=()):
    emu = emulator()
    # Tiny warmup/ramp losses must not prevent later best-model selection.
    losses = [.001] * 5 + [10., 9., 9., 9., 9., 9., 9.]
    rates = []
    for epoch, loss in enumerate(losses):
        if epoch == 0 or epoch in restarts:
            net = network()
            callbacks = emu._callbacks(patience=4, relative_improvement=relative)
            assert not emu._early_stopping_exhausted(callbacks)
            callbacks = keras.callbacks.CallbackList(callbacks, model=net)
            callbacks.on_train_begin()
        net.set_weights([np.array([[epoch]], dtype=np.float32)])
        logs = {'loss': loss, 'val_loss': loss}
        epoch_rate = float(net.optimizer.learning_rate.numpy())
        callbacks.on_epoch_end(epoch, logs)
        emu.epochs.append(epoch)
        emu.val_loss.append(loss)
        emu.learning_rate.append(epoch_rate)
        rates.append(float(net.optimizer.learning_rate.numpy()))
        if net.stop_training:
            callbacks.on_train_end()
            return epoch, rates, float(net.get_weights()[0][0, 0])
    raise AssertionError('Expected early stopping')


class SobolevStoppingScheduleTests(unittest.TestCase):
    def test_boundary_and_restart_equivalence(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                stop, rates, best_weight = run(relative)
                self.assertEqual(stop, 10)
                self.assertEqual(best_weight, 6.)
                self.assertEqual(rates[:8], [1.] * 8)
                self.assertEqual(rates[8:], [.5, .5, .25])
                resumed_stop, resumed_rates, _ = run(relative, (2, 5, 7, 9))
                self.assertEqual(resumed_stop, stop)
                self.assertEqual(resumed_rates, rates)

    def test_history_before_boundary_has_no_best_or_exhaustion(self):
        for relative in (False, True):
            emu = emulator()
            emu.epochs = list(range(5))
            emu.val_loss = [.001] * 5
            callbacks = emu._callbacks(
                patience=2, relative_improvement=relative,
                reduce_learning_rate=False)
            recovery = next(c for c in callbacks
                            if isinstance(c, EarlyStoppingHistory))
            self.assertFalse(recovery.exhausted)
            self.assertEqual(recovery.state, {})
            net = network()
            cb_list = keras.callbacks.CallbackList(callbacks, model=net)
            cb_list.on_train_begin()
            cb_list.on_epoch_end(5, {'val_loss': 10.})
            stopper = recovery.early_stopping
            self.assertEqual(stopper.best, 10.)
            self.assertEqual(stopper.wait, 0)

    def test_exhaustion_replayed_only_after_boundary(self):
        for relative in (False, True):
            emu = emulator()
            emu.epochs = list(range(8))
            emu.val_loss = [.001] * 5 + [10., 10., 10.]
            callbacks = emu._callbacks(
                patience=2, relative_improvement=relative,
                reduce_learning_rate=False)
            self.assertTrue(emu._early_stopping_exhausted(callbacks))

    def test_checkpoints_replace_provisional_warmup_best(self):
        emu = emulator()
        with tempfile.TemporaryDirectory() as path:
            callbacks = emu._callbacks(
                path=path, patience=2, reduce_learning_rate=False)
            net = network()
            net.save_weights = Mock()
            emu._save_strict_state = Mock()
            checkpoints = [c for c in callbacks if isinstance(
                c, (keras.callbacks.ModelCheckpoint, StrictStateCheckpoint))]
            for callback in checkpoints:
                callback.set_model(net)
                callback.on_train_begin()
                for epoch in range(5):
                    callback.on_epoch_end(epoch, {'val_loss': .001})
            net.save_weights.assert_called_once()
            emu._save_strict_state.assert_called_once()
            net.save_weights.reset_mock()
            emu._save_strict_state.reset_mock()
            for callback in checkpoints:
                callback.on_epoch_end(5, {'val_loss': 10.})
            net.save_weights.assert_called_once()
            emu._save_strict_state.assert_called_once()

    def test_resumed_checkpoints_use_post_ramp_best(self):
        emu = emulator()
        emu.epochs = list(range(7))
        emu.val_loss = [.001] * 5 + [10., 9.]
        with tempfile.TemporaryDirectory() as path:
            callbacks = emu._callbacks(
                path=path, patience=2, reduce_learning_rate=False)
            for callback in callbacks:
                if isinstance(callback, (keras.callbacks.ModelCheckpoint,
                                         StrictStateCheckpoint)):
                    self.assertEqual(callback.best, 9.)


if __name__ == '__main__':
    unittest.main()
