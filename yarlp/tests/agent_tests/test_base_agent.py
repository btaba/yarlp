"""
Test the base agent abstract class by instantiating a RandomAgent
"""

import unittest
import gym


from yarlp.agent.baseline_agents import RandomAgent


class TestAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('CartPole-v0')

    def test_rollout(self):
        agent = RandomAgent(self.env)
        agent.rollout()

    def test_rollout_n_steps(self):
        agent = RandomAgent(self.env)
        agent.rollout_n_steps()
