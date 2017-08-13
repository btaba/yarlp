"""
Probability distributions
"""
import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils


class Distribution(object):

    def __init__(self):
        self.sample_op = self.sample()
        self.sample_greedy_op = self.sample_greedy()

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

    def sample(self):
        raise NotImplementedError()

    def sample_greedy(self):
        raise NotImplementedError()


class Categorical(Distribution):

    def __init__(self, probs):
        self.probs = probs
        self.dist = tf.contrib.distributions.Categorical(probs=probs)
        super().__init__()

    @property
    def output_node(self):
        return self.probs

    def kl(self, old_dist):
        assert isinstance(old_dist, Categorical)
        return tf.reduce_sum(
            old_dist.probs * (
                tf.log(old_dist.probs + tf_utils.EPSILON) -
                tf.log(self.probs + tf_utils.EPSILON)
            ), axis=-1)

    def entropy(self):
        return -tf.reduce_sum(
            self.probs *
            tf.log(self.probs + tf_utils.EPSILON), axis=-1)

    def log_likelihood(self, x):
        return self.dist.log_prob(x)

    def likelihood(self, x):
        return self.dist.prob(x)

    def likelihood_ratio(self, x, old_dist):
        log_p = self.log_likelihood(x)
        return tf.exp(log_p - old_dist.log_likelihood(x))

    def sample(self):
        return tf.squeeze(self.dist.sample())

    def sample_greedy(self):
        return tf.squeeze(self.dist.mode())


class DiagonalGaussian(Distribution):

    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super().__init__()

    @property
    def output_node(self):
        return tf.squeeze(self.sample())

    def kl(self, old_dist):
        assert isinstance(old_dist, DiagonalGaussian)
        numerator = tf.square(self.std) + tf.square(self.mean - old_dist.mean)
        denominator = (2.0 * tf.square(old_dist.std))
        return tf.reduce_sum(
            old_dist.logstd - self.logstd +
            numerator / denominator - 0.5,
            axis=-1
        )

    def entropy(self):
        return tf.reduce_sum(
            self.logstd + tf.log(tf.sqrt(2. * np.pi * np.e)), axis=-1)

    def log_likelihood(self, x):
        return -1 * (0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1)\
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1])\
               + tf.reduce_sum(self.logstd, axis=-1))

    def likelihood_ratio(self, x, old_dist):
        return tf.exp(self.log_likelihood(x) - old_dist.log_likelihood(x))

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def sample_greedy(self):
        return self.mean
