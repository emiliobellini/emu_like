"""Plateau reductions and early stopping stay synchronized after resuming."""
import unittest

import numpy as np
from tensorflow import keras

from emu_like.ffnn_emu import (
    FFNNEmu, LearningRateHistory, RelativeReduceLROnPlateau,
)
from emu_like.sobolev_ffnn_emu import SobolevFFNNEmu


def scheduler(relative, patience=2, cooldown=0):
    cls = (RelativeReduceLROnPlateau if relative
           else keras.callbacks.ReduceLROnPlateau)
    kwargs = {} if relative else {'min_delta': 0.}
    return cls(patience=patience, cooldown=cooldown, mode='min',
               factor=.5, **kwargs)


def model(rate=1.):
    net = keras.Sequential([
        keras.Input(shape=(1,)), keras.layers.Dense(1),
    ])
    net.compile(optimizer=keras.optimizers.SGD(learning_rate=rate), loss='mse')
    net.stop_training = False
    return net


def replay(relative, losses, rates, patience=2, cooldown=0):
    callback = scheduler(relative, patience, cooldown)
    recovery = LearningRateHistory(
        callback, list(range(len(losses))), losses, rates)
    return callback, recovery


def run(emulator_type, relative, losses, restarts=()):
    emulator = emulator_type()
    recorded = []
    for epoch, loss in enumerate(losses):
        if epoch == 0 or epoch in restarts:
            # A strict checkpoint can contain the rate from an older best
            # epoch. Recovery must replace it with the current schedule.
            net = model()
            callbacks = emulator._callbacks(
                patience=6, relative_improvement=relative)
            if emulator._early_stopping_exhausted(callbacks):
                return recorded, epoch - 1
            callbacks = keras.callbacks.CallbackList(callbacks, model=net)
            callbacks.on_train_begin()
        logs = {'loss': loss, 'val_loss': loss}
        epoch_rate = float(net.optimizer.learning_rate.numpy())
        callbacks.on_epoch_end(epoch, logs)
        recorded.append((epoch_rate,
                         float(net.optimizer.learning_rate.numpy())))
        emulator.epochs.append(epoch)
        emulator.val_loss.append(loss)
        emulator.learning_rate.append(epoch_rate)
        if net.stop_training:
            return recorded, epoch
    return recorded, None


class LearningRateResumeTests(unittest.TestCase):
    def test_joint_schedule_matches_uninterrupted(self):
        for emulator_type in (FFNNEmu, SobolevFFNNEmu):
            for relative in (False, True):
                for losses in ([1.] * 10,
                               [1., 1.1, 1.1, .9, .91, .91, .91, .91, .91, .91],
                               [1., .9996, .9992, .9988] + [.9988] * 8):
                    with self.subTest(emulator=emulator_type, relative=relative,
                                      losses=losses):
                        expected, stop = run(emulator_type, relative, losses)
                        actual, resumed_stop = run(
                            emulator_type, relative, losses, (2, 4, 5, 7, 8))
                        np.testing.assert_allclose(actual, expected)
                        self.assertEqual(resumed_stop, stop)
        recorded, stop = run(FFNNEmu, True, [1.] * 10, (2, 4, 5))
        self.assertEqual(stop, 6)
        self.assertEqual(recorded[3], (1., .5))
        self.assertEqual(recorded[6], (.5, .25))

    def test_final_epoch_reduction_and_no_double_reduction(self):
        for relative in (False, True):
            _, recovery = replay(relative, [1.] * 3, [1.] * 3)
            self.assertEqual(recovery.next_learning_rate, .5)
            self.assertEqual(recovery.state['wait'], 0)
            _, recovery = replay(relative, [1.] * 4, [1., 1., 1., .5])
            self.assertEqual(recovery.next_learning_rate, .5)
            self.assertEqual(recovery.state['wait'], 1)

    def test_legacy_missed_reduction_is_recovered(self):
        for relative in (False, True):
            _, recovery = replay(relative, [1.] * 1434, [.0005] * 1434,
                                 patience=1000)
            self.assertAlmostEqual(recovery.next_learning_rate, .00025)
            self.assertEqual(recovery.state['wait'], 433)

    def test_cooldown_survives_restart(self):
        for relative in (False, True):
            callback, recovery = replay(relative, [1.] * 3, [1.] * 3,
                                        cooldown=3)
            net = model()
            callbacks = keras.callbacks.CallbackList([callback, recovery], model=net)
            callbacks.on_train_begin()
            self.assertEqual(callback.cooldown_counter, 3)
            callbacks.on_epoch_end(3, {'val_loss': 1.})
            self.assertEqual(callback.cooldown_counter, 2)
            self.assertEqual(float(net.optimizer.learning_rate.numpy()), .5)

    def test_warm_start_honors_selected_rate_and_restores_wait(self):
        for relative in (False, True):
            callback, recovery = replay(relative, [1.] * 4, [1., 1., 1., .5])
            FFNNEmu._configure_learning_rate_recovery([recovery], False)
            net = model(.125)
            callbacks = keras.callbacks.CallbackList([callback, recovery], model=net)
            callbacks.on_train_begin()
            self.assertEqual(float(net.optimizer.learning_rate.numpy()), .125)
            self.assertEqual(callback.wait, 1)
            callbacks.on_epoch_end(4, {'val_loss': 1.})
            self.assertEqual(float(net.optimizer.learning_rate.numpy()), .0625)

    def test_recorded_rate_increase_rebases_schedule(self):
        for relative in (False, True):
            _, recovery = replay(relative, [1.] * 6, [1., 1., 1., .5, .5, 2.])
            self.assertEqual(recovery.next_learning_rate, 2.)
            self.assertEqual(recovery.state['wait'], 1)

    def test_no_history_and_disabled_scheduler(self):
        callback, recovery = replay(True, [], [])
        self.assertIsNone(recovery.next_learning_rate)
        for cls in (FFNNEmu, SobolevFFNNEmu):
            callbacks = cls()._callbacks(patience=6, reduce_learning_rate=False)
            self.assertFalse(any(isinstance(c, LearningRateHistory) for c in callbacks))


if __name__ == '__main__':
    unittest.main()
