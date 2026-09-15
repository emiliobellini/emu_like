"""Numerical regression tests for z-only forward-mode differentiation.

Run with PYTHONPATH=src python -m unittest discover -s tests
    -p test_sobolev_derivatives.py -v
"""
import unittest

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from emu_like.sobolev_ffnn_emu import SobolevFFNNEmu, SobolevTrainingModel


class JacobianReference(SobolevTrainingModel):
    """Previous implementation, retained only as a numerical oracle."""

    def _pk_and_z_derivative(self, x_scaled, training):
        with tf.GradientTape() as tape:
            tape.watch(x_scaled)
            prediction = self.network(x_scaled, training=training)
        return prediction, tape.batch_jacobian(
            prediction, x_scaled)[:, :, self.z_index]


def copy_model(model, cls=JacobianReference):
    network = tf.keras.models.clone_model(model.network)
    network.set_weights(model.network.get_weights())
    result = cls(
        network, model.z_index, model.z_mean, model.z_scale, model.pk_scale,
        model.growth_mean, model.growth_scale, model.reference_growth,
        model.redshift_grid, model.pk_weight)
    result.fk_weight.assign(model.fk_weight)
    return result


def small_model(activation='tanh', z_index=2):
    tf.keras.utils.set_random_seed(1729)
    network = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(7, activation=activation),
        tf.keras.layers.Dense(5, activation=activation),
        tf.keras.layers.Dense(4),
    ])
    result = SobolevTrainingModel(
        network, z_index, 1., 2., [0.4, 1.3, 2., 0.7],
        [0.1, -0.2, 0.3, 0.4], [0.5, 1.2, 2., 0.7],
        [[0.8, 0.7, 0.6], [0.9, 0.8, 0.7],
         [1., 0.9, 0.8], [1.1, 1., 0.9]],
        [0., 1., 2.], 1.)
    result.fk_weight.assign(1.)
    return result


def values_and_gradients(model, x, pk, fk):
    tape, total, pk_loss, fk_loss = model._loss_terms(x, pk, fk, True)
    gradients = tape.gradient(total, model.trainable_variables)
    return (total, pk_loss, fk_loss), gradients


def physical_fk(model, x, derivative):
    z = x[:, model.z_index] * model.z_scale + model.z_mean
    return model._interpolate_reference_growth(z) - (
        0.5 * (1. + z[:, None]) * model.pk_scale / model.z_scale * derivative)


class DerivativeTests(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-6, rtol=1e-4):
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(np.isfinite(expected)))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_outputs_losses_and_weight_gradients(self):
        # Cover every supported activation and avoid assuming z is column 0.
        for activation in ['tanh', 'softplus', 'swish', 'sigmoid']:
            for z_index in [0, 2]:
                with self.subTest(activation=activation, z_index=z_index):
                    model = small_model(activation, z_index)
                    reference = copy_model(model)
                    rng = np.random.default_rng(20)
                    x = rng.uniform(-0.5, 0.5, (7, 3)).astype('float32')
                    x[:, z_index] = [-0.5, -0.4999, -0.1, 0., 0.1, 0.4999, 0.5]
                    x = tf.constant(x)
                    pk = tf.constant(rng.normal(size=(7, 4)), tf.float32)
                    fk = tf.constant(rng.normal(size=(7, 4)), tf.float32)
                    for training in [False, True]:
                        yp, dz = model._pk_and_z_derivative(x, training)
                        yr, dr = reference._pk_and_z_derivative(x, training)
                        self.assert_close(yp, yr)
                        self.assert_close(dz, dr)
                        self.assert_close(physical_fk(model, x, dz),
                                          physical_fk(reference, x, dr))
                    for pk_weight in [0., 1.]:
                        # fk alone must also propagate mixed derivatives.
                        model.pk_weight = reference.pk_weight = tf.constant(
                            pk_weight)
                        lv, gv = values_and_gradients(model, x, pk, fk)
                        lr, gr = values_and_gradients(reference, x, pk, fk)
                        self.assert_close(lv, lr)
                        for a, b in zip(gv, gr):
                            self.assertEqual(a is None, b is None)
                            if a is not None:
                                self.assert_close(a, b)

    def test_graph_dynamic_batch_and_sample_independence(self):
        model = small_model()
        reference = copy_model(model)
        fn = tf.function(model._pk_and_z_derivative,
                         input_signature=[tf.TensorSpec([None, 3], tf.float32),
                                          tf.TensorSpec([], tf.bool)],
                         jit_compile=False)
        x = tf.constant([[0.1, 0.2, -0.5], [0.3, -0.2, 0.5]])
        for batch in [x, x[:1]]:
            result = fn(batch, False)
            expected = reference._pk_and_z_derivative(batch, False)
            for a, b in zip(result, expected):
                self.assert_close(a, b)
        self.assert_close(fn(x, False)[1][:1], fn(x[:1], False)[1])

    def test_float64_finite_differences(self):
        # Independent network-only check, including a weight derivative of
        # a loss that depends on dz. All layers explicitly use float64.
        tf.keras.utils.set_random_seed(12)
        net = tf.keras.Sequential([
            tf.keras.Input(shape=(3,), dtype='float64'),
            tf.keras.layers.Dense(5, activation='tanh', dtype='float64'),
            tf.keras.layers.Dense(4, dtype='float64'),
        ])
        model = small_model()
        model.network = net
        x = tf.constant([[0.2, 0.3, -0.1], [0.4, -0.2, 0.5]], tf.float64)
        v = tf.broadcast_to(tf.constant([0., 0., 1.], tf.float64), tf.shape(x))
        _, dz = model._pk_and_z_derivative(x, False)
        for h in [1e-3, 1e-4, 1e-5]:
            finite = (net(x + h*v) - net(x - h*v)) / (2*h)
            self.assert_close(dz, finite, atol=2e-7, rtol=2e-5)

        def loss():
            y, d = model._pk_and_z_derivative(x, True)
            return tf.reduce_mean(y*y + d*d)

        with tf.GradientTape() as tape:
            value = loss()
        grads = tape.gradient(value, net.trainable_variables)
        rng = np.random.default_rng(99)
        directions = [
            rng.normal(size=w.shape) for w in net.trainable_variables]
        norm = np.sqrt(sum(np.sum(d*d) for d in directions))
        directions = [d / norm for d in directions]
        exact = sum(np.sum(g.numpy()*d) for g, d in zip(grads, directions))
        weights = net.get_weights()
        for h in [1e-3, 1e-4, 1e-5]:
            net.set_weights([w+h*d for w, d in zip(weights, directions)])
            plus = float(loss())
            net.set_weights([w-h*d for w, d in zip(weights, directions)])
            minus = float(loss())
            net.set_weights(weights)
            self.assert_close(exact, (plus-minus)/(2*h), atol=2e-7, rtol=2e-5)

    def test_eval_fk_physical_inputs(self):
        emu = SobolevFFNNEmu()
        emu.model = small_model()
        emu.z_index = 2
        emu.x_names = ['a', 'b', 'z_pk']
        emu.x_scaler = StandardScaler().fit([[-1., -1., -1.], [1., 1., 3.]])
        x = np.array([[0.2, 0.1, 0.], [-0.3, 0.4, 2.]], 'float32')
        scaled = tf.constant(emu.x_scaler.transform(x), tf.float32)
        reference = copy_model(emu.model)
        _, derivative = reference._pk_and_z_derivative(scaled, False)
        expected = physical_fk(reference, scaled, derivative)
        self.assert_close(emu.eval_fk(x), expected)
        self.assert_close(emu.eval_fk(x[0]), expected[0])
        self.assert_close(
            emu.eval_fk(dict(zip(emu.x_names, x[0]))), expected[0])

    def test_analytic_redshift_scaling(self):
        model = small_model()
        model.network = tf.keras.Sequential([
            tf.keras.Input(shape=(3,)), tf.keras.layers.Dense(4)])
        kernel = np.array([[0.2, 0.1, -0.3, 0.8], [1., -1., 0.5, 0.2],
                           [1., 2., -0.5, 0.3]], 'float32')
        model.network.set_weights([kernel, np.zeros(4, 'float32')])
        x = tf.constant([[0., 0., -0.5], [0.2, 0.4, 0.], [0., 1., 0.5]])
        _, derivative = model._pk_and_z_derivative(x, False)
        self.assert_close(derivative, np.broadcast_to(kernel[2], (3, 4)))
        z = np.array([0., 1., 2.])
        reference = np.array([[0.8, 0.9, 1., 1.1],
                              [0.7, 0.8, 0.9, 1.],
                              [0.6, 0.7, 0.8, 0.9]])
        expected = reference - 0.5*(1.+z[:, None]) * (
            np.array([0.4, 1.3, 2., 0.7])/2. * kernel[2])
        self.assert_close(physical_fk(model, x, derivative), expected)

    def test_optimizer_updates_and_zero_weight_branch(self):
        model = small_model()
        reference = copy_model(model)
        for m in [model, reference]:
            m.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3), jit_compile=False)
        x = np.random.default_rng(1).uniform(-0.5, 0.5, (7, 3)).astype(
            'float32')
        y = {
            'pk': np.ones((7, 4), 'float32'),
            'fk': np.zeros((7, 4), 'float32')}
        # Exercise the actual tf.cond path across the warm-up transition.
        for weight in [0., 0.001, 1., 1., 1.]:
            model.fk_weight.assign(weight)
            reference.fk_weight.assign(weight)
            model.reset_metrics()
            reference.reset_metrics()
            a = model.train_on_batch(x, y, return_dict=True)
            b = reference.train_on_batch(x, y, return_dict=True)
            for name in a:
                self.assert_close(a[name], b[name])
            for wa, wb in zip(model.network.weights,
                              reference.network.weights):
                self.assert_close(wa.numpy(), wb.numpy())
            for va, vb in zip(model.optimizer.variables,
                              reference.optimizer.variables):
                self.assert_close(va.numpy(), vb.numpy())


if __name__ == '__main__':
    unittest.main()
