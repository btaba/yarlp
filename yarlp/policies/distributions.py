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

    def likelihood_ratio(self, x, old_output):
        raise NotImplementedError()

    def log_likelihood(self, x):
        raise NotImplementedError()

    def likelihood(self, x):
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
        # assert isinstance(old_prob, Categorical)
        return tf.reduce_sum(
            old_prob.output * (
                tf.log(old_prob.output + tf_utils.EPSILON) -
                tf.log(self.output + tf_utils.EPSILON)
            ), axis=-1)

    def entropy(self):
        return -tf.reduce_sum(
            self.output *
            tf.log(self.output + tf_utils.EPSILON), axis=-1)

    def log_likelihood(self, x):
        p = self.likelihood(x)
        log_p = tf.squeeze(tf.log(p + tf_utils.EPSILON))
        return log_p

    def likelihood(self, x):
        x_onehot = tf.one_hot(x, self.output_node.shape[1])
        p = tf.reduce_sum(x_onehot * self.output, 1)
        return p

    def likelihood_ratio(self, x, old_logli):
        log_p = self.log_likelihood(x)
        return tf.exp(log_p - old_logli)


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
        # assert isinstance(old_prob, DiagonalGaussian)
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
        n = tf.contrib.distributions.Normal(self.mean, self.std)
        return n.log_prob(x)

    def likelihood(self, x):
        n = tf.contrib.distributions.Normal(self.mean, self.std)
        return n.prob(x)

    def likelihood_ratio(self, x, old_logli):
        return tf.exp(self.log_likelihood(x) - old_logli)
