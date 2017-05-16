"""
    CEM Agent class, which takes in a PolicyModel object
"""
import os
from functools import partial
from yarlp.agent.base_agent import Agent
from yarlp.model.model import Model
from yarlp.utils.env_utils import GymEnv

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
    policy_model : model.Model

    num_samples : integer
        Total number of sample weights to draw for each training step

    init_var : float, default 0.1

    best_pct : float, default 0.2
        The percentage of sample weights to keep that yield the best reward
    """
    def __init__(self, env, num_samples,
                 init_var=0.1, best_pct=0.2,
                 policy_network=tf.contrib.layers.fully_connected,
                 model_file_path=None,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._policy = CEMAgent.policy_model_factory(
            env, policy_network, model_file_path)

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

    def save_models(self, path):
        path = os.path.join(path, 'cem_policy')
        self._policy.save(path)

    @staticmethod
    def policy_model_factory(env, policy_network, model_file_path):
        """ Network for CEM agents
        """

        def build_graph(model, env, network):
            model.add_input()

            if GymEnv.env_action_space_is_discrete(env):
                network = partial(network, activation_fn=tf.nn.softmax)
                model.add_output(network)
            else:
                network = partial(network, activation_fn=None)
                model.add_output(network, clip_action=True)

        def build_update_feed_dict(model):
            pass

        build_graph = partial(build_graph, env=env, network=policy_network)

        if model_file_path is not None:
            return Model(env, None, build_update_feed_dict,
                         os.path.join(model_file_path, 'cem_policy'))
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

    def train(self, num_train_steps, num_test_steps=0,
              with_variance=False, alpha=5, beta=10,
              min_sigma=0.01):
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
        for i in range(num_train_steps):
            # logger.info('Training step {}'.format(i))
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
                rollout = self.rollout()
                rollout_rewards.append(np.sum(rollout.rewards))

            # get the num_best mean/var with highest reward
            rollout_rewards = np.array(rollout_rewards)
            self.logger.set_metrics_for_rollout(rollout, train=True)
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
