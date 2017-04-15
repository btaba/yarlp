"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import unittest
import gym

from yarlp.agent.cem_agent import CEMAgent


class TestCEMCartPole(unittest.TestCase):
    """
    Cart-pole regression test on CEM agent
    """

    def test_cem(self):
        env = gym.make('CartPole-v0')
        agent = CEMAgent(
            env,
            num_samples=25, init_var=.1, best_pct=0.2)
        agent.train(num_train_steps=1, with_variance=True)
