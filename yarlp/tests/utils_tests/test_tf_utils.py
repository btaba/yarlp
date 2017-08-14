import unittest
import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils


class TestTfUtils(unittest.TestCase):

    def test_set_global_seed(self):
        tf_utils.set_global_seeds(0)
        tf_u_gen = tf.random_uniform([1])

        u = np.random.uniform()
        with tf.Session() as sess1:
            tf_u = sess1.run(tf_u_gen)

        tf_utils.set_global_seeds(0)
        u2 = np.random.uniform()
        with tf.Session() as sess2:
            tf_u2 = sess2.run(tf_u_gen)

        assert np.isclose(u, u2)
        assert np.allclose(tf_u, tf_u2)
