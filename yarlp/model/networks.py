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


def dense(x, size, name, weight_init=None, bias=True, activation_fn=None):
    w = tf.get_variable(name + "/weight", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/bias", [size], initializer=tf.zeros_initializer())
        out = ret + b
    else:
        out = ret

    if activation_fn is not None:
        return activation_fn(out)
    return out


def mlp(inputs, num_outputs, final_activation_fn=None,
        activation_fn=None, hidden_units=(32, 32),
        weights_initializer=normc_initializer(1.0),
        final_weights_initializer=normc_initializer(0.01)):
    """
    Multi-Layer Perceptron
    """

    if activation_fn is None:
        activation_fn = tf.nn.tanh

    assert len(hidden_units) > 0

    if isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    assert isinstance(hidden_units, tuple)

    x = inputs
    for i, h in enumerate(hidden_units):
        x = dense(x, h, 'fully_connected{}'.format(i),
                  weight_init=weights_initializer,
                  activation_fn=activation_fn)

    x = dense(x, num_outputs, 'fully_connected_final',
              activation_fn=final_activation_fn,
              weight_init=final_weights_initializer)
    
    return x
