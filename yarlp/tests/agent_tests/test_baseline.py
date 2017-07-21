"""
    Regression tests for baselines
"""

import unittest
import gym

from yarlp.agent.baseline_agents import RandomAgent


class TestBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('CartPole-v0')

    def test_random(self):
        agent = RandomAgent(self.env)
        agent.train(num_train_steps=1)
