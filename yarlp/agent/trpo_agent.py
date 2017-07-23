"""
TRPO
"""

import os
import numpy as np
import tensorflow as tf

from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import discrete_trpo_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline
from yarlp.utils.experiment_utils import get_network
from yarlp.utils.env_utils import GymEnv


class TRPOAgent(Agent):
    """
    TRPO

    Parameters
    ----------
    env : gym.env

    policy_network : model.Model

    policy_learning_rate : float, the learning rate for the policy_network

    baseline_network : if None, we us no baseline
        otherwise we use a LinearFeatureBaseline as default
        you can also pass in a function as a tensorflow network which
        gets built by the value_function_model_factory

    model_file_path : str, file path for the policy_network
    """
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 policy_learning_rate=0.01,
                 baseline_network=LinearFeatureBaseline(),
                 baseline_model_learning_rate=0.01,
                 model_file_path=None,
                 adaptive_std=False,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        policy_path = None
        if model_file_path:
            policy_path = os.path.join(model_file_path, 'pg_policy')

        policy_network = get_network(policy_network, policy_network_params)

        if GymEnv.env_action_space_is_discrete(env):
            self._policy = discrete_trpo_model_factory(
                env, policy_network, policy_learning_rate,
                model_file_path=policy_path)
        else:
            raise NotImplementedError()

        self._baseline_model = baseline_network

    def save_models(self, path):
        ppath = os.path.join(path, 'pg_policy')
        self._policy.save(ppath)

    def train(self, num_train_steps=10, num_test_steps=0,
              n_steps=1000, render=False, whiten_advantages=True):
        """

        Parameters
        ----------
        num_train_steps : integer
            Total number of training iterations.

        num_test_steps : integer
            Number of testing iterations per training iteration.

        n_steps : integer
            Total number of samples from the environment for each
            training iteration.

        whiten_advantages : bool, whether to whiten the advantages

        render : bool, whether to render episodes in a video

        Returns
        ----------
        None
        """

        # config
        cg_damping = 1e-5   # reg coeff
        max_kl = 1e-2   # step size

        for i in range(num_train_steps):
            # execute an episode
            rollouts = self.rollout_n_steps(n_steps, render=render)

            actions = []
            action_probs = []
            states = []
            advantages = []
            discounted_rewards = []

            for rollout in rollouts:
                discounted_reward = self.get_discounted_reward_list(
                    rollout.rewards)

                baseline_pred = np.zeros_like(discounted_reward)
                if self._baseline_model:
                    baseline_pred = self._baseline_model.predict(
                        np.array(rollout.states)).flatten()

                baseline_pred = np.append(baseline_pred, 0)
                advantage = rollout.rewards + self._discount *\
                    baseline_pred[1:] - baseline_pred[:-1]
                advantage = self.get_discounted_reward_list(
                    advantage)

                advantages = np.concatenate([advantages, advantage])
                states.append(rollout.states)
                actions.append(rollout.actions)
                action_probs.append(rollout.action_probs)
                discounted_rewards = np.concatenate(
                    [discounted_rewards, discounted_reward])

            states = np.concatenate([s for s in states]).squeeze()
            actions = np.concatenate([a for a in actions])
            action_probs = np.concatenate([a for a in action_probs]).squeeze()

            if whiten_advantages:
                advantages = (advantages - np.mean(advantages)) /\
                    (np.std(advantages) + 1e-8)

            # batch update the baseline model
            if self._baseline_model:
                self._baseline_model.fit(states, discounted_rewards)

            # update the policy
            feed = self._policy.build_update_feed_dict(
                self._policy,
                states, advantages.squeeze(), actions.squeeze(),
                action_probs.squeeze())
            thprev = self._policy.G(self._policy.gf, feed)

            def fisher_vector_product(p):
                feed[self._policy.flat_tangent] = p
                return self._policy.G(self._policy.fvp, feed) + cg_damping * p

            g = self._policy.G(self._policy.pg, feed)
            if np.allclose(g, np.zeros_like(g)):
                print('Gradient zero, skipping update.')
                continue

            # descent direciton
            stepdir = conjugate_gradient(fisher_vector_product, g)
            initial_step_size = np.sqrt(
                2.0 * max_kl *
                (1. / (stepdir.dot(fisher_vector_product(stepdir)) + 1e-8))
            )

            print('initial_step_size is ', initial_step_size)
            if np.isnan(initial_step_size):
                print('NAN initial_step_size')
                initial_step_size = 1.
            flat_descent_step = initial_step_size * stepdir

            def get_loss(th):
                feed[self._policy.theta] = th
                self._policy.G(self._policy.sff, feed)
                return self._policy.G(self._policy.losses[0], feed)
            lossbefore = get_loss(thprev)

            backtrack_ratio = 0.8
            max_backtracks = 15
            for n_iter, ratio in\
                    enumerate(backtrack_ratio ** np.arange(max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_theta = thprev - cur_step

                # set the theta and compute the loss
                feed[self._policy.theta] = cur_theta
                self._policy.G(self._policy.sff, feed)
                loss = get_loss(cur_theta)
                constraint_val = self._policy.G(
                    self._policy.kl, feed)
                if loss < lossbefore and constraint_val <= max_kl:
                    break
            if (np.isnan(loss) or np.isnan(constraint_val) or
                    loss >= lossbefore or constraint_val >= max_kl):
                print("Line search condition violated. Rejecting the step!")
                if np.isnan(loss):
                    print("Violated because loss is NaN")
                if np.isnan(constraint_val):
                    print("Violated because constraint is NaN")
                if loss >= lossbefore:
                    print("Violated because loss not improving")
                if constraint_val >= max_kl:
                    print("Violated because constraint is violated")
                get_loss(thprev)
            else:
                assert not np.isclose(
                    (self._policy.G(self._policy.theta, feed) -
                     thprev).sum(), 0.)

            surrafter, kloldnew, entropy = self._policy.G(
                self._policy.losses, feed_dict=feed)

            print("Entropy", entropy)
            print("KL between old and new distribution", kloldnew)
            print("Surrogate loss before", lossbefore)
            print("Surrogate loss after", surrafter)

            self.logger.set_metrics_for_rollout(rollouts, train=True)
            self.logger.log()

            # if num_test_steps > 0:
            #     r = []
            #     for t_test in range(num_test_steps):
            #         rollout = self.rollout(greedy=True)
            #         r.append(rollout)
            #     self.logger.add_metric('policy_loss', 0)
            #     self.logger.set_metrics_for_rollout(r, train=False)
            #     self.logger.log()

            # if self.logger._log_dir is not None:
            #     self.save_models(self.logger._log_dir)

        return


def conjugate_gradient(f_Ax, b, cg_iters=10,
                       callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    if np.any(np.isnan(b)):
        print('THERE are NANS in b!')
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
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

    if np.any(np.isnan(x)):
        print('THERE are NANS in x!')
    return x


def test_cg():
    A = np.random.randn(5, 5)
    A = A.T.dot(A)
    b = np.random.randn(5)
    x = conjugate_gradient(lambda x: A.dot(x), b, cg_iters=5, verbose=True)  # pylint: disable=W0108
    assert np.allclose(A.dot(x), b)


def linesearch(f, x, fullstep, expected_improve_rate):
    backtrack_ratio = 0.8
    accept_ratio = .1
    max_backtracks = 15
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(backtrack_ratio ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x
