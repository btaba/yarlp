"""
    CEM Agent class, which takes in a PolicyModel object

[1] Learning Tetris with the Noisy Cross-Entropy Method
    (Szita, Lorincz 2006)
    pdf: http://nipg.inf.elte.hu/publications/szita06learning.pdf
"""

from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import cem_model_factory

import numpy as np
import tensorflow as tf


class CEMAgent(Agent):
    """
    Cross Entropy Method

    where Z_t = max(alpha - t / beta) from equation (2.5)

    Parameters
    ----------
    policy_model : model.Model

    n_weight_samples : integer
        Total number of sample weights to draw during each training step

    init_var : float, default 0.1

    best_pct : float, default 0.2
        The percentage of sample weights to keep that yield the best reward
    """
    def __init__(self, env, n_weight_samples=100,
                 init_var=1., best_pct=0.2,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 model_file_path=None,
                 min_std=1e-6, init_std=1.0, adaptive_std=False,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self._policy = cem_model_factory(
            env, policy_network, policy_network_params,
            min_std=min_std, init_std=init_std,
            adaptive_std=adaptive_std,
            model_file_path=model_file_path)

        # Get model shapes
        theta = self._policy.G(self._policy.gf)
        self.model_total_sizes = theta.shape[0]

        # Mean and SD of our weights
        self._theta = np.zeros(self.model_total_sizes)
        self._sigma = np.ones(self.model_total_sizes) * init_var

        # Total number of sample weights to draw for each training step
        self.n_weight_samples = n_weight_samples

        # Number of best parameters to keep
        assert best_pct <= 1 and best_pct > 0
        self.num_best = int(best_pct * self.n_weight_samples)

    def save_models(self, path):
        self._policy.save(path)

    def train(self, num_train_steps, num_test_steps=0,
              with_variance=False, alpha=5, beta=10,
              min_sigma=1e-2, render=False):
        """
        Learn the most optimal weights for our PolicyModel
            optionally with a variance adjustment as in [1]
            as Z_t = max(alpha - t / beta)

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        with_variance : boolean
            train with or without a variance adjustment as in [1]

        alpha : float, default 5
            parameter for variance adjustment (Z_t = max(alpha - t / beta))

        beta : float, default 10
            parameter for variance adjustment (Z_t = max(alpha - t / beta))

        render : bool, whether to render episodes in a video

        Returns
        ----------
        None
        """
        for i in range(num_train_steps):
            # logger.info('Training step {}'.format(i))
            if np.any(self._sigma <= 0):
                self._sigma[self._sigma <= 0] = min_sigma

            # generate n_samples each iteration with new mean and stddev
            # according to our current optimal mean and variance
            weight_samples = np.array(
                [np.random.normal(mean, np.sqrt(var), self.n_weight_samples)
                 for mean, var in zip(self._theta, self._sigma)])
            weight_samples = weight_samples.T

            # get the rewards for each mean/variance
            rollouts = []
            rollout_rewards = []
            for w in weight_samples:
                # add weights to PolicyModel
                self._policy.G(
                    self._policy.sff,
                    {self._policy.theta: w})
                rollout = self.rollout(render=render)
                rollouts.append(rollout)
                rollout_rewards.append(np.sum(rollout.rewards))

            # get the num_best mean/var with highest reward
            rollout_rewards = np.array(rollout_rewards)
            self.logger.set_metrics_for_rollout(rollouts, train=True)
            self.logger.log()

            best_idx = rollout_rewards.argsort()[::-1][:self.num_best]
            best_samples = weight_samples[best_idx]

            # recompute our mean/var of our weights
            mean = best_samples.mean(axis=0)
            var = best_samples.var(axis=0)

            if with_variance:
                var += max(alpha - i / float(beta), 0)

            self._theta = mean
            self._sigma = var

            if num_test_steps > 0:
                r = []
                for t_test in range(num_test_steps):
                    rollout = self.rollout(greedy=True)
                    r.append(rollout)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()

            if self.logger._log_dir is not None:
                self.save_models(self.logger._log_dir)

        return
