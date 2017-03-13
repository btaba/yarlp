"""
    REINFORCE Agent class, which takes in a PolicyModel object
"""

from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import value_function_model_factory

import numpy as np
import tensorflow as tf


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient
    and Policy Gradient with Function Approximation
    [1] Simple statistical gradient-following algorithms for connectionist
        reinforcement learning (Williams, 1992)
        pdf: http://link.springer.com/article/10.1007/BF00992696

    [2] Sutton, R. S., Mcallester, D., Singh, S., & Mansour, Y. (1999).
        Policy Gradient Methods for Reinforcement Learning with
        Function Approximation. Advances in Neural Information Processing
        Systems 12, 1057–1063. doi:10.1.1.37.9714

    Parameters
    ----------
    policy_model : tf_model.Model

    value_model : either 'linear' or 'average', defaults to 'average'

    """
    def __init__(self, policy_model, value_model='average',
                 value_model_learning_rate=0.01, *args, **kwargs):
        super().__init__(policy_model._env, *args, **kwargs)
        self._policy = policy_model

        if value_model == 'linear':
            # Policy Gradient with function approximation
            # theoretically the state-value approximation should
            # be linear in policy features if the policy model is a softmax
            # due to Theorem 2 in [2]
            self._value_model = value_function_model_factory(
                self._policy.env, network=tf.contrib.layers.fully_connected,
                learning_rate=value_model_learning_rate)
        else:
            # Classic REINFORCE from [1]
            self._value_model = None

    def train(self, num_training_steps, with_baseline=True):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        Returns
        ----------
        total_reward_per_training_episode : list
            total reward obtained after each training episode

        """
        total_reward_per_training_episode = []
        for i in range(num_training_steps):
            # execute an episode
            rollout = self.rollout()

            # save average reward for this training step for reporting
            total_reward_per_training_episode.append(np.sum(rollout.rewards))

            for t, r in enumerate(rollout.rewards):
                # update the weights for policy model
                discounted_rt = self.get_discounted_cumulative_reward(
                    rollout.rewards[t:])

                baseline = 0
                if with_baseline:
                    if self._value_model:
                        self._value_model.update(
                            rollout.states[t], discounted_rt)
                        baseline = self._value_model.predict(
                            np.array(rollout.states[t])).flatten()[0]
                    else:
                        baseline = np.mean(rollout.rewards[t:])

                advantage = discounted_rt - baseline
                self._policy.update(
                    rollout.states[t], advantage, rollout.actions[t])

        return total_reward_per_training_episode


class OneStepActorCriticPG(Agent):
    """One-step actor critic with policy gradients.
    Boostrapping returns introduces bias and can be difficult to tune.

    Parameters
    ----------
    """

    def __init__(self, policy_model,
                 value_model_learning_rate=0.1, *args, **kwargs):
        super().__init__(policy_model._env, *args, **kwargs)
        self._policy = policy_model

        self._value_model = value_function_model_factory(
            self._policy.env, network=tf.contrib.layers.fully_connected,
            learning_rate=value_model_learning_rate)

    def train(self, num_training_steps, with_baseline=True):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        Returns
        ----------
        total_reward_per_training_episode : list
            total reward obtained after each training episode

        """
        total_reward_per_training_episode = []
        for i in range(num_training_steps):
            # execute an episode
            total_rewards = 0
            obs = self._env.reset()
            for t in range(self.num_max_rollout_steps):
                action = self.get_action(obs)
                (obs_prime, reward, done, _) = self._env.step(action)

                total_rewards += reward

                v_prime = 0 if done else self._value_model.predict(
                    np.array(obs_prime)).flatten()[0]
                v = self._value_model.predict(np.array(obs)).flatten()[0]

                one_step_td_target = reward + self._discount * v_prime
                td_error = one_step_td_target - v

                # the state-value function should model the one-step TD reward
                self._value_model.update(obs, one_step_td_target)

                # one-step TD error is advantage in policy update
                self._policy.update(
                    obs, td_error * (self._discount ** t), action)
                obs = obs_prime

                if done:
                    break
            print(t, total_rewards)

            total_reward_per_training_episode.append(total_rewards)

        return total_reward_per_training_episode
