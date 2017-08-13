"""
Different networks
"""
import numpy as np
import tensorflow as tf


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def mlp(inputs, num_outputs, final_activation_fn=tf.nn.softmax,
        activation_fn=tf.nn.tanh, hidden_units=(32, 32),
        weights_initializer=normc_initializer(1.0),
        final_weights_initializer=normc_initializer(0.01)):
    """
    Multi-Layer Perceptron
    """
    assert len(hidden_units) > 0

    if isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    assert isinstance(hidden_units, tuple)

    x = inputs
    for h in hidden_units:
        x = tf.contrib.layers.fully_connected(
            x, num_outputs=h, activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer())

    x = tf.contrib.layers.fully_connected(
        x, num_outputs=num_outputs,
        activation_fn=final_activation_fn,
        weights_initializer=final_weights_initializer,
        biases_initializer=tf.zeros_initializer())

    return x
