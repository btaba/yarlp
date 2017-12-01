"""
TRPO
"""

import numpy as np
import tensorflow as tf

from yarlp.agent.base_agent import BatchAgent
from yarlp.model.model_factories import trpo_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline
from yarlp.utils.experiment_utils import get_network
from yarlp.model.model_factories import value_function_model_factory


class TRPOAgent(BatchAgent):
    """
    TRPO

    Parameters
    ----------
    env : gym.env

    policy_network : model.Model

    baseline_network : if None, we us no baseline
        otherwise we use a LinearFeatureBaseline as default
        you can also pass in a function as a tensorflow network which
        gets built by the value_function_model_factory

    model_file_path : str, file path for the policy_network
    """

    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 baseline_network=None,
                 baseline_model_learning_rate=0.001,
                 model_file_path=None,
                 adaptive_std=False,
                 gae_lambda=0.98, cg_iters=10,
                 cg_damping=1e-1, max_kl=1e-2,
                 input_shape=None,
                 init_std=1.0, min_std=1e-6,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        policy_network = get_network(policy_network, policy_network_params)

        self._policy = trpo_model_factory(
            env, network=policy_network, network_params=policy_network_params,
            min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
            input_shape=input_shape, model_file_path=model_file_path)

        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger._logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self._gae_lambda = gae_lambda

        if isinstance(baseline_network, LinearFeatureBaseline):
            self._baseline_model = baseline_network
        elif baseline_network is None:
            self._baseline_model = LinearFeatureBaseline()
        else:
            self._baseline_model = value_function_model_factory(
                env, policy_network,
                learning_rate=baseline_model_learning_rate)

    def save_models(self, path):
        self._policy.save(path)

    def update(self, path):
        # update the policy
        feed = self._policy.build_update_feed_dict(
            self._policy,
            path['states'], path['advantages'],
            path['actions'])
        thprev = self._policy.G(self._policy.gf, feed)
        self._policy.G(self._policy.set_old_pi_eq_new_pi)

        fvp_feed = self._policy.build_update_feed_dict(
            self._policy,
            path['states'][::5], path['advantages'][::5],
            path['actions'][::5])

        def fisher_vector_product(p):
            fvp_feed[self._policy.flat_tangent] = p
            return self._policy.G(self._policy.fvp, fvp_feed) +\
                self.cg_damping * p

        g = self._policy.G(self._policy.pg, feed)
        if np.allclose(g, 0):
            print('Gradient zero, skipping update.')
            return

        # descent direciton
        stepdir = conjugate_gradient(fisher_vector_product, g, self.cg_iters)

        def get_loss(th):
            feed[self._policy.theta] = th
            self._policy.G(self._policy.sff, feed)
            return self._policy.G(self._policy.losses[0], feed)
        lossbefore = get_loss(thprev)

        assert np.isfinite(stepdir).all()
        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        expectedimprove = g.dot(fullstep)
        surrbefore = lossbefore
        stepsize = 1.0

        for _ in range(10):
            thnew = thprev - fullstep * stepsize  # plus or minus?

            surr = get_loss(thnew)
            kl = self._policy.G(self._policy.kl, feed)

            improve = surrbefore - surr
            print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
            if not np.isfinite(surr).all():
                print("Got non-finite value of losses -- bad!")
            elif kl > self.max_kl * 1.5:
                print("violated KL constraint. shrinking step.")
            elif improve < 0:
                print("surrogate didn't improve. shrinking step.")
            else:
                print("Stepsize OK!")
                break
            stepsize *= .5
        else:
            print("couldn't compute a good step")
            get_loss(thprev)

        surrafter, kloldnew, entropy = self._policy.G(
            self._policy.losses, feed_dict=feed)

        print("Entropy", entropy)
        print("KL between old and new distribution", kloldnew)
        print("Surrogate loss before", lossbefore)
        print("Surrogate loss after", surrafter)

        return


def conjugate_gradient(f_Ax, b, cg_iters=10, callback=None,
                       verbose=True, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)

    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))

    return x


def test_cg():
    A = np.random.randn(5, 5)
    A = A.T.dot(A)
    b = np.random.randn(5)
    x = conjugate_gradient(lambda x: A.dot(x), b, cg_iters=5, verbose=True)
    assert np.allclose(A.dot(x), b)


def linesearch(f, x, fullstep, expected_improve_rate):
    backtrack_ratio = 0.8
    accept_ratio = .1
    max_backtracks = 15
    fval = f(x)

    for (_n_backtracks, stepfrac) in \
            enumerate(backtrack_ratio ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x
