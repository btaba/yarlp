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


def mlp(inputs, num_outputs, final_activation_fn=None,
        activation_fn=None, hidden_units=[32, 32],
        weights_initializer=normc_initializer(1.0),
        final_weights_initializer=normc_initializer(0.01),
        final_scope='action',
        *args, **kwargs):
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
        x = tf.layers.dense(
            x, h, kernel_initializer=weights_initializer,
            activation=activation_fn
        )

    with tf.variable_scope(final_scope):
        x = tf.layers.dense(
            x, num_outputs, activation=final_activation_fn,
            kernel_initializer=final_weights_initializer
        )

    return x


def cnn(*args, **kwargs):
    return cnn2(*args, **kwargs)


def cnn2(inputs, num_outputs,
         convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
         hidden_units=[512],
         final_activation_fn=None,
         dueling=False,
         final_dense_weights_initializer=tf.contrib.layers.xavier_initializer(),
         padding='same',
         final_scope='action'):

    if isinstance(final_dense_weights_initializer, float):
        final_dense_weights_initializer = normc_initializer(
            final_dense_weights_initializer)

    with tf.variable_scope('cnn'):
        x = inputs
        for n_out, kernel_size, strides in convs:
            x = tf.layers.conv2d(
                x, filters=n_out,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

        x = tf.layers.flatten(x)

        with tf.variable_scope(final_scope):
            x_a = x
            for h in hidden_units:
                x_a = tf.layers.dense(
                    x_a, h, activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            x_a = tf.layers.dense(
                x_a, num_outputs, activation=final_activation_fn,
                kernel_initializer=final_dense_weights_initializer)

        output = x_a
        if dueling:
            x_s = x
            for h in hidden_units:
                x_s = tf.layers.dense(
                    x_s, h, activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            x_s = tf.layers.dense(
                x_s, 1, activation=final_activation_fn,
                kernel_initializer=final_dense_weights_initializer)
            x_a_mean = tf.reduce_mean(x_a, axis=1)
            x_a_centered = x_a - tf.expand_dims(x_a_mean, 1)
            output = x_s + x_a_centered

    return output
