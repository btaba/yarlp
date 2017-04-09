"""
    CEM Agent class, which takes in a PolicyModel object
"""
from functools import partial
from yarlp.agent.base_agent import Agent
from yarlp.model.tf_model import Model
from yarlp.utils.env_utils import env_action_space_is_discrete
from yarlp.utils.logger import logger

import numpy as np
import tensorflow as tf


class CEMAgent(Agent):
    """
    Cross Entropy Method
    [1] Learning Tetris with the Noisy Cross-Entropy Method
        (Szita, Lorincz 2006)
        pdf: http://nipg.inf.elte.hu/publications/szita06learning.pdf

    where Z_t = max(alpha - t / beta) from equation (2.5)

    Parameters
    ----------
    policy_model : tf_model.Model

    num_samples : integer
        Total number of sample weights to draw for each training step

    init_var : float, default 0.1

    best_pct : float, default 0.2
        The percentage of sample weights to keep that yield the best reward
    """
    def __init__(self, env, num_samples,
                 init_var=0.1, best_pct=0.2,
                 policy_network=tf.contrib.layers.fully_connected,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._policy = CEMAgent.policy_model_factory(env, policy_network)

        # Get model shapes
        self.model_shapes = [w.shape for w in self._policy.get_weights()]
        self.model_sizes = [w.size for w in self._policy.get_weights()]
        self.model_total_sizes = sum(self.model_sizes)

        # Mean and SD of our weights
        self._theta = np.zeros(self.model_total_sizes)
        self._sigma = np.ones(self.model_total_sizes) * init_var

        # Total number of sample weights to draw for each training step
        self.num_samples = num_samples

        # Number of best parameters to keep
        assert best_pct <= 1 and best_pct > 0
        self.num_best = int(best_pct * self.num_samples)

        logger.info('CEMAgent created.')

    @staticmethod
    def policy_model_factory(env, policy_network):
        """ Network for CEM agents
        """

        def build_graph(model, env, network):
            model.add_input()

            if env_action_space_is_discrete(env):
                network = partial(network, activation_fn=tf.nn.softmax)
                model.add_output(network)
            else:
                network = partial(network, activation_fn=None)
                model.add_output(network, clip_action=True)

        def build_update_feed_dict(model):
            pass

        build_graph = partial(build_graph, env=env, network=policy_network)

        return Model(env, build_graph, build_update_feed_dict)

    @property
    def theta(self):
        """
        Returns
        ----------
        np.array of weights of policy model
        """
        return self._reshape_weights(self._theta)

    @property
    def sigma(self):
        """
        Returns
        ----------
        np.array of standard deviation of weights of policy model
        """
        return self._reshape_weights(self._sigma)

    def train(self, num_training_steps, with_variance=False, alpha=5, beta=10,
              min_sigma=0.01, render=False, render_freq=5):
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

        Returns
        ----------
        None
        """
        for i in range(num_training_steps):
            logger.info('Training step {}'.format(i))
            if np.any(self._sigma <= 0):
                # warnings.warn(
                #     ("Variance for at least one weight "
                #         "is less than or equal to 0"),
                #     Warning)
                # return total_reward_per_training_episode
                self._sigma[self._sigma <= 0] = min_sigma

            # generate n_samples each iteration with new mean and stddev
            # according to our current optimal mean and variance
            weight_samples = np.array(
                [np.random.normal(mean, np.sqrt(var), self.num_samples)
                 for mean, var in zip(self._theta, self._sigma)])
            weight_samples = weight_samples.T

            # get the rewards for each mean/variance
            rollout_rewards = []
            for w in weight_samples:
                # add weights to PolicyModel
                weights_reshaped = self._reshape_weights(w)
                self._policy.set_weights(weights_reshaped)
                rollout = self.rollout(render=render, render_freq=render_freq)
                rollout_rewards.append(np.sum(rollout.rewards))

            # get the num_best mean/var with highest reward
            rollout_rewards = np.array(rollout_rewards)
            logger.info(
                'Average rollout rewards: {}'.format(np.mean(rollout_rewards)))
            logger.info(
                'Total rollout rewards: {}'.format(np.sum(rollout_rewards)))

            best_idx = rollout_rewards.argsort()[::-1][:self.num_best]
            best_samples = weight_samples[best_idx]

            # recompute our mean/var of our weights
            mean = best_samples.mean(axis=0)
            var = best_samples.var(axis=0)

            if with_variance:
                var += max(alpha - i / float(beta), 0)

            self._theta = mean
            self._sigma = var

        return

    def _reshape_weights(self, weights_flat):
        """
        Reshape weights from flat array to
        the shape needed by model object

        Parameters
        ----------
        weights_flat : numpy array

        Returns
        ----------
        weights : weights reshaped according to our model shapes
        """
        p = 0
        weights = []
        for idx, size in enumerate(self.model_sizes):
            array = weights_flat[p:(p + size)]
            array = array.reshape(self.model_shapes[idx])
            weights.append(array)
            p += size
        return weights
