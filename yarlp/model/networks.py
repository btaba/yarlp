"""
Different networks
"""
import tensorflow as tf


VSI = tf.contrib.layers.variance_scaling_initializer()


def mlp(inputs, num_outputs, activation_fn=tf.nn.softmax,
        hidden_units=(10, 10)):

    x = tf.contrib.layers.fully_connected(
        inputs, num_outputs=hidden_units[0])

    for h in hidden_units[1:]:
        x = tf.contrib.layers.fully_connected(
            x, num_outputs=h)

    x = tf.contrib.layers.fully_connected(
        x, num_outputs=num_outputs,
        activation_fn=activation_fn)
    return x
