# -*- coding: utf-8 -*-
"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.model.softmax_model import TFSoftmaxModel
from yarlp.agent.reinforce_agent import REINFORCEAgent


class TestREINFORCECartPole(unittest.TestCase):
    """
    """
    @classmethod
    def setUpClass(cls):
        env = gym.make('CartPole-v0')
        cls.lm_tf = TFSoftmaxModel(env)

    def test_reinforce(self):
        agent = REINFORCEAgent(
            self.lm_tf, num_training_steps=100, num_max_rollout_steps=1000,
            discount=.95)
        agent.train()

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)
