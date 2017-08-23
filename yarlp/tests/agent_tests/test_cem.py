"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import unittest
import gym
# import shutil

from yarlp.agent.cem_agent import CEMAgent


class TestCEMCartPole(unittest.TestCase):
    """
    Cart-pole regression test on CEM agent
    """

    def test_cem(self):
        env = gym.make('CartPole-v0')
        agent = CEMAgent(
            env,
            n_weight_samples=25, init_var=.1, best_pct=0.2)
        agent.train(num_train_steps=1, with_variance=True)

    # def test_cem_save_model(self):
    #     env = gym.make('CartPole-v0')
    #     agent = CEMAgent(
    #         env,
    #         n_weight_samples=25, init_var=.1, best_pct=0.2)
    #     agent.save_models('testy_cem')

    #     agent = CEMAgent(env, n_weight_samples=25,
    #                      init_var=.1, best_pct=.2, model_file_path='testy_cem')

    # def tearDown(self):
    #     shutil.rmtree('testy_cem')
