# -*- coding: utf-8 -*-
"""
    CEM Agent class, which takes in a PolicyModel object
"""
from yarlp.agent.base_agent import Agent

import warnings
import numpy as np


class CEMAgent(Agent):
    """
    Cross Entropy Method
    [1] Learning Tetris with the Noisy Cross-Entropy Method
        (Szita, Lorincz 2006)
        pdf: http://nipg.inf.elte.hu/publications/szita06learning.pdf

    where Z_t = max(alpha - t / beta) from equation (2.5)

    Parameters
    ----------
    policy_model : PolicyModel object where the model is a keras model object

    num_training_steps : integer
        Total number of training steps

    num_max_rollout_steps : integer
        Maximum number of steps in an episode

    num_samples : integer
        Total number of sample weights to draw for each training step

    init_var : float, default 0.1

    best_pct : float, default 0.2
        The percentage of sample weights to keep that yield the best reward
    """
    def __init__(self, policy_model, num_training_steps,
                 num_max_rollout_steps, num_samples,
                 init_var=0.1, best_pct=0.2):
        super().__init__(policy_model._env)
        self._policy = policy_model

        # Get model shapes
        self.model_shapes = [w.shape for w in self._policy.get_weights()]
        self.model_sizes = [w.size for w in self._policy.get_weights()]
        self.model_total_sizes = sum(self.model_sizes)

        # Mean and SD of our weights
        self._theta = np.zeros(self.model_total_sizes)
        self._sigma = np.ones(self.model_total_sizes) * init_var

        # Maximum number of steps in an episode
        self.num_max_rollout_steps = num_max_rollout_steps

        # Total number of training steps
        self.num_training_steps = num_training_steps

        # Total number of sample weights to draw for each training step
        self.num_samples = num_samples

        # Number of best parameters to keep
        assert best_pct <= 1 and best_pct > 0
        self.num_best = int(best_pct * self.num_samples)

        # Number of actions to take
        self.num_actions = self._env.action_space.n

    @property
    def policy(self):
        return self._policy

    @property
    def theta(self):
        """
        Returns
        ----------
        np.array of weights of policy model
        """
        return self.reshape_weights(self._theta)

    @property
    def sigma(self):
        """
        Returns
        ----------
        np.array of standard deviation of weights of policy model
        """
        return self.reshape_weights(self._sigma)

    def train(self, with_variance=False, alpha=5, beta=10):
        """
        Learn the most optimal weights for our PolicyModel
            optionally with a variance adjustment as in [1]
            as Z_t = max(alpha - t / beta)

        Parameters
        ----------
        with_variance : boolean
            train with or without a variance adjustment as in [1]

        alpha : float, default 5
            parameter for variance adjustment (Z_t = max(alpha - t / beta))

        beta : float, default 10
            parameter for variance adjustment (Z_t = max(alpha - t / beta))

        Returns
        ----------
        avg_reward_per_training_step : list
            average reward obtained on each training step

        """
        avg_reward_per_training_step = []
        for i in range(self.num_training_steps):
            if np.any(self._sigma <= 0):
                # raise ValueError(
                #     'Variance for weight is less than or equal to 0')
                warnings.warn(
                    ("Variance for at least one weight "
                        "is less than or equal to 0"),
                    Warning)
                return avg_reward_per_training_step

            # generate n_samples each iteration with new mean and stddev
            # according to our current optimal mean and variance
            weight_samples = np.array(
                [np.random.normal(mean, np.sqrt(var), self.num_samples)
                 for mean, var in zip(self._theta, self._sigma)])
            weight_samples = weight_samples.T

            # get the rewards for each mean/variance
            rollout_rewards = np.array(
                [self.rollout(w)[0] for w in weight_samples])

            # save average reward for this training step
            avg_reward_per_training_step.append(np.mean(rollout_rewards))

            # get the num_best mean/var with highest reward
            best_idx = rollout_rewards.argsort()[::-1][:self.num_best]
            best_samples = weight_samples[best_idx]

            # recompute our mean/var of our weights
            mean = best_samples.mean(axis=0)
            var = best_samples.var(axis=0)

            if with_variance:
                var += max(alpha - i / beta, 0)

            self._theta = mean
            self._sigma = var

        return avg_reward_per_training_step

    def rollout(self, weights):
        """
        Performs actions for num_max_rollout_steps on the environment
        based on the agent's current weights

        Parameters
        ----------
        weights : numpy array of model weights, shape = ??

        Returns
        ----------
        total_reward : float
        t + 1 : integer
            number of time steps completed during rollout
        """

        # add weights to PolicyModel
        weights_reshaped = self.reshape_weights(weights)
        self._policy.set_weights(weights_reshaped)

        # do rollout
        total_reward = 0
        observation = self._env.reset()
        for t in range(self.num_max_rollout_steps):
            action = self.get_action(observation)
            (observation, reward, done, _) = self._env.step(action)
            total_reward += reward
            if done:
                break

        return total_reward, t + 1

    def do_greedy_episode(self, max_time_steps=1000):
        t = 0
        done = False
        total_reward = 0
        observation = self._env.reset()
        while not done and t < max_time_steps:
            action = self.get_action(observation, greedy=True)
            (observation, reward, done, _) = self._env.step(action)
            total_reward += reward
            t += 1
        return total_reward

    def reshape_weights(self, weights_flat):
        """
        Reshape weights from flat array of numbers to
        the shape needed by keras model object

        Parameters
        ----------
        weights_flat : numpy array, shape = ??

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

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        integer indicating the action that should be taken
        """
        batch = np.array([state])
        action = self._policy.predict_on_batch(batch).flatten()
        if not greedy:
            return np.random.choice(np.arange(self.num_actions), p=action)
        return self.argmax_break_ties(action)

    def argmax_break_ties(self, probs):
        """
        Breaks ties randomly in an array of probabilities

        Parameters
        ----------
        probs : numpy array, shape = (1, ?)

        Returns
        ----------
        integer indicating the action that should be taken
        """
        return np.random.choice(np.where(probs == probs.max())[0])
