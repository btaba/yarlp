# -*- coding: utf-8 -*-
"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.model.model_factories import cem_model_factory
from yarlp.agent.cem_agent import CEMAgent


class TestCEMCartPole(unittest.TestCase):
    """
    Cart-pole regression test on CEM agent
    """
    @classmethod
    def setUpClass(cls):
        env = gym.make('CartPole-v0')
        cls.lm_tf = cem_model_factory(env)

    def test_cem_tf(self):
        # To solve the Cart-Pole we must get avg reward > 195
        # over 100 consecutive trials
        agent = CEMAgent(
            self.lm_tf, num_max_rollout_steps=1000,
            num_samples=10, init_var=.1, best_pct=0.2)
        agent.train(num_training_steps=50, with_variance=True)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)
