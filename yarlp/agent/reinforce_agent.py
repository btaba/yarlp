# -*- coding: utf-8 -*-
"""
    REINFORCE Agent class, which takes in a PolicyModel object
"""

from yarlp.agent.base_agent import Agent

import warnings
import numpy as np


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient
    [1] Simple statistical gradient-following algorithms for connectionist
        reinforcement learning (Williams, 1992)
        pdf: http://link.springer.com/article/10.1007/BF00992696

    Parameters
    ----------
    policy_model : PolicyModel object where the model is a keras model object


    """
    def __init__(self, policy_model,
                 num_training_steps,
                 num_max_rollout_steps,
                 discount=1):
        super().__init__(policy_model._env)
        self._policy = policy_model

        # Discount factor
        self._discount = 1

        # Get model shapes
        self.model_shapes = [w.shape for w in self._policy.get_weights()]
        self.model_sizes = [w.size for w in self._policy.get_weights()]
        self.model_total_sizes = sum(self.model_sizes)

        # Maximum number of steps in an episode
        self.num_max_rollout_steps = num_max_rollout_steps

        # Total number of training steps
        self.num_training_steps = num_training_steps

        # Number of actions that can be taken on each step
        self.num_actions = self._env.action_space.n

    def train(self, with_baseline=False):
        """

        Parameters
        ----------
        with_baseline : boolean
            train with or without a baseline value estimator which reduces
            variance in policy gradient updates

        Returns
        ----------
        avg_reward_per_training_step : list
            average reward obtained on each training step

        """
        avg_reward_per_training_step = []
        for i in range(self.num_training_steps):
            # execute an episode
            rollout_rewards, rollout_actions, rollout_states = self.rollout()

            # save average reward for this training step for reporting
            avg_reward_per_training_step.append(np.mean(rollout_rewards))
            print('%d Reward is: ' % (i), sum(rollout_rewards))

            for t, r in enumerate(rollout_rewards):
                # update the weights for policy model
                discounted_rt = self.get_discounted_cumulative_reward(
                    rollout_rewards[t:])
                self._policy.update(
                    rollout_states[t], discounted_rt, rollout_actions[t])

        return avg_reward_per_training_step

    def rollout(self):
        """
        Performs actions for num_max_rollout_steps on the environment
        based on the agent's current weights

        Parameters
        ----------

        Returns
        ----------
        total_reward : float
        t + 1 : integer
            number of time steps completed during rollout
        """

        rewards = []
        states = []
        actions = []
        observation = self._env.reset()
        for t in range(self.num_max_rollout_steps):
            states.append(observation)
            action = self.get_action(observation)
            (observation, reward, done, _) = self._env.step(action)
            rewards.append(reward)
            actions.append(action)
            if done:
                break

        return rewards, actions, states

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

    def get_discounted_cumulative_reward(self, rewards):
        """
        Parameters
        ----------
        r : list

        Returns
        ----------
        cumulative_reward : list
        """
        cumulative_reward = [0]
        for t, r in enumerate(rewards):
            temp = cumulative_reward[-1] + self._discount ** t * r
            cumulative_reward.append(temp)

        return np.sum(cumulative_reward[1:])

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
