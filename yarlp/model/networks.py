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
        x = tf.layers.dense(
            x, h, kernel_initializer=weights_initializer,
            activation=activation_fn
        )

    x = tf.layers.dense(
        x, num_outputs, activation=final_activation_fn,
        kernel_initializer=final_weights_initializer
    )

    return x


# def flattenallbut0(x):
#     # return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])
#     return tf.layers.flatten(x)


# def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
#            summary_tag=None):
#     with tf.variable_scope(name):
#         stride_shape = [1, stride[0], stride[1], 1]
#         filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

#         # there are "num input feature maps * filter height * filter width"
#         # inputs to each hidden unit
#         fan_in = intprod(filter_shape[:3])
#         # each unit in the lower layer receives a gradient from:
#         # "num output feature maps * filter height * filter width" /
#         #   pooling size
#         fan_out = intprod(filter_shape[:2]) * num_filters
#         # initialize weights with random weights
#         w_bound = np.sqrt(6. / (fan_in + fan_out))

#         w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
#                             collections=collections)
#         b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
#                             collections=collections)

#         if summary_tag is not None:
#             tf.summary.image(summary_tag,
#                              tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
#                                           [2, 0, 1, 3]),
#                              max_images=10)

#         return tf.nn.conv2d(x, w, stride_shape, pad) + b


def cnn(inputs, num_outputs,
        dense_weights_initializer=normc_initializer(1.0),
        final_weights_initializer=normc_initializer(0.01),
        final_activation_fn=None):

    # x = tf.nn.relu(U.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
    # x = tf.nn.relu(U.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
    # x = U.flattenallbut0(x)
    # x = tf.nn.relu(U.dense(x, 128, 'lin', U.normc_initializer(1.0)))
    # logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))

    x = inputs
    print(x)
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

    # x = tf.nn.relu(conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
    # x = tf.nn.relu(conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
    # x = flattenallbut0(x)
    x = tf.reshape(x, [-1, np.product(x.get_shape().as_list()[1:])])

    x = tf.layers.dense(
        x, 128, activation=tf.nn.relu,
        kernel_initializer=dense_weights_initializer
    )

    # x = tf.nn.relu(
    #     dense(x, 128, 'lin', dense_weights_initializer))

    x = tf.layers.dense(
        x, num_outputs,
        activation=final_activation_fn,
        kernel_initializer=final_weights_initializer
    )
    # self.vpred = dense(x, num_outputs, "value", final_dense_weights_initializer)

    return x
