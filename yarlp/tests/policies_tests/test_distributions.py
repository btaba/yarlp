import pytest
import numpy as np
import tensorflow as tf
import scipy.stats as stats
from yarlp.policy.distributions import Categorical, DiagonalGaussian


def test_diag_gauss_ent_and_kl():
    np.random.seed(1)
    N = 200000

    # diagonal gaussian
    mean = np.array([[-.2, .3, .4, -.5]], dtype='float32')
    logstd = np.array([[.1, -.5, .1, 0.8]], dtype='float32')
    mean2 = mean * np.random.randn(mean.shape[-1]) * 0.1
    logstd2 = logstd * np.random.randn(mean.shape[-1]) * 0.1

    means = np.vstack([mean] * N)
    logstds = np.vstack([logstd] * N)
    dist = DiagonalGaussian({}, means, logstds)
    q_dist = DiagonalGaussian(
        {}, mean2.astype('float32'), logstd2.astype('float32'))
    validate_probtype(dist, q_dist, N)


def test_categorical_ent_and_kl():
    np.random.seed(1)
    N = 200000

    # categorical
    logit = np.array([[.2, .3, .5]], dtype='float32')
    logits = np.vstack([logit] * N)
    dist = Categorical({}, logits)
    output2 = logit + np.random.rand(logit.shape[-1]) * .1
    q_dist = Categorical({}, output2.astype('float32'))
    validate_probtype(dist, q_dist, N)


def test_diag_gauss_against_scipy():
    sess = tf.Session()
    mean = np.array([[-.2, .3, .4, -.5]], dtype='float32')
    logstd = np.array([[.1, -.5, .1, 0.8]], dtype='float32')
    dist = DiagonalGaussian({}, mean, logstd)

    # validate log likelihood
    n = stats.multivariate_normal(
        mean=mean[0], cov=np.square(np.diag(np.exp(logstd[0]))))
    x = np.array([[0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]], dtype='float32')
    assert np.allclose(n.logpdf(x), sess.run(dist.log_likelihood(x)))

    # validate entropy
    assert np.isclose(n.entropy(), sess.run(dist.entropy()))


def test_categorical_against_scipy():
    sess = tf.Session()
    logits = np.array([[.2, .3, .5]], dtype='float32')
    dist = Categorical({}, logits)

    probs = np.exp(logits) / np.exp(logits).sum()
    c = stats.multinomial(p=probs, n=1)

    assert np.allclose(sess.run(dist.probs), c.p)

    # validate log likelihood
    x = np.array([[1], [2], [0]])
    x_one_hot = np.zeros((3, 3))
    x_one_hot[np.arange(3), x.flatten()] = 1
    assert np.allclose(
        sess.run(dist.log_likelihood(x)).squeeze(),
        c.logpmf(x_one_hot))

    # validate entropy
    assert np.isclose(c.entropy()[0], sess.run(dist.entropy())[0])


def validate_probtype(dist, q_dist, N):
    """
    Test copied from openai/baselines
    """

    # Check to see if mean negative log likelihood == differential entropy
    # sample X from the distribution
    sess = tf.Session()
    Xval = sess.run(dist.sample())
    Xval = np.array(Xval)
    print(Xval)
    # get the mean negative log likelihood for sampled X
    negloglik = -1 * dist.log_likelihood(Xval)
    negloglik = sess.run(negloglik)
    print(negloglik)
    # assert that the mean negative log likelihood is within
    # 3 standard errors of the entropy
    ent = sess.run(dist.entropy()).mean()
    assert abs(ent - negloglik.mean()) < 3 * negloglik.std() / np.sqrt(N),\
        str((ent, negloglik.mean(), negloglik.std() / np.sqrt(N)))

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    kl = sess.run(dist.kl(q_dist)).mean()
    loglik = sess.run(q_dist.log_likelihood(Xval))
    kl_ll = -ent - loglik.mean()
    kl_ll_stderr = loglik.std() / np.sqrt(N)
    assert np.abs(kl - kl_ll) < 3 * kl_ll_stderr,\
        str((kl, kl_ll, kl_ll_stderr))
