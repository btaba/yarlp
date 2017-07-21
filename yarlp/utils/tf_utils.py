"""
Tensorflow util functions
"""

import numpy as np
import tensorflow as tf

EPSILON = 1e-8


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return flatten_vars(grads)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out),\
        "shape function assumes that shape is fully known"
    return out


def flatten_vars(var_list):
    return tf.concat(
        [tf.reshape(v, [-1]) for v in var_list], axis=0)


def setfromflat(var_list, theta):
    assigns = []
    shapes = map(var_shape, var_list)
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = np.prod(shape)
        assigns.append(
            tf.assign(
                v,
                tf.reshape(
                    theta[
                        start:start +
                        size],
                    shape)))
        start += size
    return tf.group(*assigns)
