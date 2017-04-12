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
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=1)

    def test_reinforce_w_baseline(self):
        agent = REINFORCEAgent(
            self.env,
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=1)
