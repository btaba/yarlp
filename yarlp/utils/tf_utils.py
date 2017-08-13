"""
Tensorflow util functions
"""

import numpy as np
import tensorflow as tf

EPSILON = 1e-8
_CACHED_PLACEHOLDER = {}

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


def get_placeholder(name, dtype, shape):
    if name in _CACHED_PLACEHOLDER:
        out, dtype1, shape1 = _CACHED_PLACEHOLDER[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _CACHED_PLACEHOLDER[name] = (out, dtype, shape)
        return out
