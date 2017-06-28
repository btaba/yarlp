"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import shutil

from yarlp.agent.pg_agents import REINFORCEAgent


class TestREINFORCECartPole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('CartPole-v0')

    def test_reinforce_no_baseline(self):
        agent = REINFORCEAgent(
            self.env, baseline_model=None,
            discount_factor=.95)
        agent.train(num_train_steps=1)

    def test_reinforce_w_baseline(self):
        agent = REINFORCEAgent(
            self.env,
            discount_factor=.95)
        agent.train(num_train_steps=1)

    def test_reinforce_save_models(self):
        agent = REINFORCEAgent(
            self.env,
            discount_factor=.95)
        agent.save_models('testy_reinforce')
        agent = REINFORCEAgent(
            self.env,
            model_file_path='testy_reinforce')
        shutil.rmtree('testy_reinforce')
