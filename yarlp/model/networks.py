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

    x = tf.layers.dense(
        x, num_outputs, activation=final_activation_fn,
        kernel_initializer=final_weights_initializer
    )

    return x


def cnn(*args, **kwargs):
    # return cnn2(*args, **kwargs)
    return nature_cnn(*args, **kwargs)


def cnn1(inputs, num_outputs,
         dense_weights_initializer=normc_initializer(1.0),
         final_weights_initializer=normc_initializer(0.01),
         final_activation_fn=None):

    x = inputs

    x = tf.layers.conv2d(
        x, filters=8,
        kernel_size=(8, 8),
        strides=(4, 4),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )

    x = tf.layers.conv2d(
        x, filters=16,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )

    x = tf.reshape(x, [-1, np.product(x.get_shape().as_list()[1:])])
    # x = tf.layers.flatten(x)

    x = tf.layers.dense(
        x, 128, activation=tf.nn.relu,
        kernel_initializer=dense_weights_initializer
    )

    x = tf.layers.dense(
        x, num_outputs,
        activation=final_activation_fn,
        kernel_initializer=final_weights_initializer
    )

    return x


def cnn2(inputs, num_outputs,
         convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
         hidden_units=[512],
         final_activation_fn=None,
         dueling=False,
         final_dense_weights_initializer=tf.contrib.layers.xavier_initializer(),
         padding='same'):

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

        with tf.variable_scope('action'):
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


from yarlp.external.baselines.baselines.a2c.utils import conv, fc, conv_to_fc

count = 0

def nature_cnn(inputs, num_outputs, final_dense_weights_initializer, **kwargs):
    """
    CNN from Nature paper.
    """
    global count
    scaled_images = inputs
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    z = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
    count += 1
    a = fc(z, str(count), num_outputs, init_scale=final_dense_weights_initializer)
    return a