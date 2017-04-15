"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym

from yarlp.agent.pg_agents import REINFORCEAgent


class TestREINFORCECartPole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('CartPole-v0')

    def test_reinforce_no_baseline(self):
        agent = REINFORCEAgent(
            self.env, baseline_network=None,
            discount_factor=.95)
        agent.train(num_train_steps=1)

    def test_reinforce_w_baseline(self):
        agent = REINFORCEAgent(
            self.env,
            discount_factor=.95)
        agent.train(num_train_steps=1)
