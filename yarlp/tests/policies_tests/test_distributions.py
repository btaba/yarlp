import unittest
import numpy as np
import tensorflow as tf
from yarlp.policy.distributions import Categorical, DiagonalGaussian


class TestDistributions(unittest.TestCase):

    def test_dists(self):
        np.random.seed(0)

        # diagonal gaussian
        mean = np.array([[-.2, .3, .4, -.5]], dtype='float32')
        logstd = np.array([[.1, -.5, .1, 0.8]], dtype='float32')
        dist = DiagonalGaussian(mean, logstd)
        mean2 = mean * np.random.randn(mean.shape[0]) * 0.1
        logstd2 = logstd * np.random.randn(mean.shape[0]) * 0.1
        q_dist = DiagonalGaussian(
            mean2.astype('float32'), logstd2.astype('float32'))
        validate_probtype(dist, q_dist)

        # categorical
        probs = np.array([[.2, .3, .5]], dtype='float32')
        dist = Categorical(probs)
        output2 = probs + np.random.rand(probs.shape[0]) * .1
        output2 /= output2.sum()
        q_dist = Categorical(output2.astype('float32'))
        validate_probtype(dist, q_dist)


def validate_probtype(dist, q_dist):
    """
    Test copied from openai/baselines
    """
    N = 100000

    # Check to see if mean negative log likelihood == differential entropy
    # sample X from the distribution
    sess = tf.Session()
    Xval = sess.run(dist.sample(N))
    Xval = np.array(Xval)
    # get the mean negative log likelihood for sampled X
    negloglik = -1 * dist.log_likelihood(Xval)
    negloglik = sess.run(negloglik)
    # assert that the mean negative log likelihood is within
    # 3 standard errors of the entropy
    ent = sess.run(dist.entropy())
    assert abs(ent[0] - negloglik.mean()) < 3 * negloglik.std() / np.sqrt(N)

    # same test using likelihood instead of log_likelihood function
    negloglik = -1 * tf.log(dist.likelihood(Xval))
    negloglik = sess.run(negloglik)
    assert abs(ent[0] - negloglik.mean()) < 3 * negloglik.std() / np.sqrt(N)

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    kl = sess.run(dist.kl(q_dist))[0]
    loglik = sess.run(q_dist.log_likelihood(Xval))
    kl_ll = -ent - loglik.mean()
    kl_ll_stderr = loglik.std() / np.sqrt(N)
    assert np.abs(kl - kl_ll[0]) < 3 * kl_ll_stderr
