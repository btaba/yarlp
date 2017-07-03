"""
Different networks
"""
import tensorflow as tf


def mlp(inputs, num_outputs, activation_fn=tf.nn.softmax,
        hidden_units=(10, 10),
        weights_initializer=tf.contrib.layers.xavier_initializer()):
    """
    Multi-Layer Perceptron
    """
    assert len(hidden_units) > 0

    if isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    assert isinstance(hidden_units, tuple)

    x = tf.contrib.layers.fully_connected(
        inputs, num_outputs=hidden_units[0],
        weights_initializer=weights_initializer)

    for h in hidden_units[1:]:
        x = tf.contrib.layers.fully_connected(
            x, num_outputs=h,
            weights_initializer=weights_initializer)

    x = tf.contrib.layers.fully_connected(
        x, num_outputs=num_outputs,
        activation_fn=activation_fn,
        weights_initializer=weights_initializer)
    return x
