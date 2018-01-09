"""
Probability distributions
"""
import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils


class Distribution(object):

    def __init__(self, model):
        self.model[self.scope + ':sample_op'] = self.sample()
        self.model[self.scope + ':sample_greedy_op'] = self.sample_greedy()

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

    def __init__(self, model, logits):
        self.model = model
        self.scope = scope = tf.get_variable_scope().name
        model[scope + ':logits'] = logits
        logits1 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
        exp_logits = tf.exp(logits1)
        Z = tf.reduce_sum(exp_logits, axis=-1, keep_dims=True)
        model[scope + ':probs'] = exp_logits / Z
        model[scope + ':log_probs'] = logits1 - tf.log(Z)
        super().__init__(model)

    @property
    def output_node(self):
        return tf.squeeze(self.sample())

    @property
    def logits(self):
        return self.model[self.scope + ':logits']

    @property
    def probs(self):
        return self.model[self.scope + ':probs']

    @property
    def log_probs(self):
        return self.model[self.scope + ':log_probs']

    def kl(self, old_dist):
        probs = self.probs
        old_probs = old_dist.probs
        return tf.reduce_sum(
            old_probs * (
                tf.log(old_probs + tf_utils.EPSILON) -
                tf.log(probs + tf_utils.EPSILON)
            ), axis=-1)

    def entropy(self):
        probs = self.probs
        return -tf.reduce_sum(
            probs * tf.log(probs + tf_utils.EPSILON), axis=-1)

    def log_likelihood(self, x):
        x_one_hot = tf.one_hot(tf.squeeze(x), depth=tf.shape(self.logits)[-1])
        return tf.reduce_sum(self.log_probs * x_one_hot, axis=-1)

    def likelihood_ratio(self, x, old_dist):
        log_p = self.log_likelihood(x)
        return tf.exp(log_p - old_dist.log_likelihood(x))

    def sample(self):
        # Gumbel max trick for sampling in log-space
        logits = self.logits
        u = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

    def sample_greedy(self):
        return tf.argmax(self.logits, axis=-1)


class DiagonalGaussian(Distribution):

    def __init__(self, model, mean, logstd):
        self.model = model
        self.scope = scope = tf.get_variable_scope().name
        model[scope + ':mean'] = mean
        model[scope + ':logstd'] = logstd
        model[scope + ':std'] = tf.exp(logstd)
        super().__init__(model)

    @property
    def std(self):
        return self.model[self.scope + ':std']

    @property
    def mean(self):
        return self.model[self.scope + ':mean']

    @property
    def logstd(self):
        return self.model[self.scope + ':logstd']

    @property
    def output_node(self):
        return tf.squeeze(self.sample())

    def kl(self, old_dist):
        numerator = tf.square(self.std) + tf.square(self.mean - old_dist.mean)
        denominator = (2.0 * tf.square(old_dist.std))
        return tf.reduce_sum(
            old_dist.logstd - self.logstd +
            numerator / denominator - 0.5,
            axis=-1
        )

    def entropy(self):
        return tf.reduce_sum(
            0.5 * tf.log(2. * np.pi * np.e) + self.logstd, axis=-1)

    def log_likelihood(self, x):
        dim = tf.to_float(tf.shape(self.mean)[-1])
        log_det_cov = 2. * tf.reduce_sum(self.logstd, axis=-1)
        dev = x - self.mean
        maha = tf.reduce_sum(tf.square(dev / self.std), axis=-1)
        return -0.5 * (dim * tf.log(2 * np.pi) + log_det_cov + maha)

    def likelihood_ratio(self, x, old_dist):
        return tf.exp(self.log_likelihood(x) - old_dist.log_likelihood(x))

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def sample_greedy(self):
        return self.mean
