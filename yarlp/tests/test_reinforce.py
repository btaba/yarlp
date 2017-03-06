# -*- coding: utf-8 -*-
"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.model.model_factories import policy_gradient_model_factory
from yarlp.model.model_factories import value_function_model_factory
from yarlp.agent.reinforce_agent import REINFORCEAgent


class TestREINFORCECartPole(unittest.TestCase):
    """
    """
    @classmethod
    def setUpClass(cls):
        env = gym.make('CartPole-v0')
        cls.lm_tf = policy_gradient_model_factory(env)
        cls.value = value_function_model_factory(env, learning_rate=0.01)

    def test_reinforce(self):
        # To solve the Cart-Pole we must get avg reward > 195
        # over 100 consecutive trials
        agent = REINFORCEAgent(
            self.lm_tf, num_max_rollout_steps=1000,
            discount_factor=.95)
        agent.train(num_training_steps=100, with_baseline=False)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(np.mean(sampled_greedy_rewards))
        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)

    def test_reinforce_w_baseline(self):
        agent = REINFORCEAgent(
            self.lm_tf, self.value, num_max_rollout_steps=1000,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(np.mean(sampled_greedy_rewards))
        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)
