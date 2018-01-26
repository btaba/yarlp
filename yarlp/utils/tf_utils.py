"""
Tensorflow util functions
"""

import random
import numpy as np
import tensorflow as tf

EPSILON = 1e-8
_CACHED_PLACEHOLDER = {}


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


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


def reset_cache():
    global _CACHED_PLACEHOLDER
    _CACHED_PLACEHOLDER = {}


def get_placeholder(name, dtype, shape):
    if name in _CACHED_PLACEHOLDER:
        out, dtype1, shape1 = _CACHED_PLACEHOLDER[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _CACHED_PLACEHOLDER[name] = (out, dtype, shape)
        return out


def iterbatches(arrays, batch_size=64):
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:]
    for batch_inds in np.array_split(inds, sections):
        if len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
