# -*- coding: utf-8 -*-
"""
    REINFORCE Agent class, which takes in a PolicyModel object
"""

from yarlp.agent.base_agent import Agent

import numpy as np


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient
    [1] Simple statistical gradient-following algorithms for connectionist
        reinforcement learning (Williams, 1992)
        pdf: http://link.springer.com/article/10.1007/BF00992696

    Parameters
    ----------
    policy_model : tf_model.Model

    value_model : tf_model.value_model


    """
    def __init__(self, policy_model, value_model=None, *args, **kwargs):
        super().__init__(policy_model._env, *args, **kwargs)
        self._policy = policy_model
        self._value_model = value_model

    def train(self, num_training_steps, with_baseline=True):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        Returns
        ----------
        avg_reward_per_training_step : list
            average reward obtained on each training step

        """

        if with_baseline:
            assert self._value_model is not None,\
                "Must specify value function model to train with baseline."

        avg_reward_per_training_step = []
        for i in range(num_training_steps):
            # execute an episode
            rollout = self.rollout()

            # save average reward for this training step for reporting
            avg_reward_per_training_step.append(np.mean(rollout.rewards))
            print('%d Reward is: ' % (i), sum(rollout.rewards))

            for t, r in enumerate(rollout.rewards):
                # update the weights for policy model
                discounted_rt = self.get_discounted_cumulative_reward(
                    rollout.rewards[t:])

                baseline = 0
                if with_baseline:
                    self._value_model.update(rollout.states[t], discounted_rt)
                    baseline = self._value_model.predict(
                        np.array(rollout.states[t])).flatten()[0]

                advantage = discounted_rt - baseline
                self._policy.update(
                    rollout.states[t], advantage, rollout.actions[t])

        return avg_reward_per_training_step
