"""
Probability distributions
"""
import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils


class Distribution(object):

    def kl(self, other):
        raise NotImplementedError()

    @property
    def output_node(self):
        raise NotImplementedError()

    def entropy(self):
        raise NotImplementedError()

    def likelihood_ratio(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError()

    def log_likelihood(self, x):
        raise NotImplementedError()

    def sample():
        raise NotImplementedError()


class Categorical(Distribution):

    def __init__(self, output):
        self.output = output

    @property
    def output_node(self):
        return self.output

    def kl(self, old_prob):
        return tf.reduce_sum(
            old_prob * (
                tf.log(old_prob + tf_utils.EPSILON) -
                tf.log(self.output + tf_utils.EPSILON)
            ), axis=-1)

    def entropy(self):
        return -tf.reduce_sum(
            self.output *
            tf.log(self.output + tf_utils.EPSILON), axis=-1)

    def log_likelihood(self, x):
        x_onehot = tf.one_hot(x, self.output_node.shape[1])
        p = tf.reduce_sum(x_onehot * self.output, 1)
        log_p = tf.squeeze(tf.log(p + tf_utils.EPSILON))
        return log_p

    def likelihood_ratio(self, x, old_output):
        log_p = self.log_likelihood(x)
        # old_log_p = tf.squeeze(
        #     tf.log(tf.reduce_sum(x * old_output, 1) + tf_utils.EPSILON))
        old_log_p = old_output.log_likelihood(x)
        return tf.exp(log_p - old_log_p)


class DiagonalGaussian(Distribution):

    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    @property
    def output_node(self):
        n = tf.contrib.distributions.Normal(self.mean, self.std)
        return tf.squeeze(n.sample([1]))

    def kl(self, old_prob):
        assert isinstance(old_prob, DiagonalGaussian)
        numerator = tf.square(old_prob.mean - self.mean) +\
            tf.square(old_prob.std) - tf.square(self.std)
        denominator = 2 * tf.square(self.std) + tf_utils.EPSILON
        return tf.reduce_sum(
            numerator / denominator + self.logstd - old_prob.logstd,
            axis=-1)

    def entropy(self):
        return tf.reduce_sum(
            self.logstd + tf.log(tf.sqrt(2. * np.pi * np.e)), axis=-1)

    def log_likelihood(self, x):
        zs = (x - self.mean) / self.std
        return -tf.reduce_sum(self.logstd, axis=len(x.get_shape()) - 1) -\
            0.5 * tf.reduce_sum(tf.square(zs), axis=len(x.get_shape()) - 1) -\
            0.5 * tf.to_float(tf.shape(x)[-1]) * np.log(2 * np.pi)

    def likelihood_ratio(self, x, old_output):
        return tf.exp(self.log_likelihood(x) - old_output.log_likelihood(x))
