"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import pytest
import gym
# import shutil

from yarlp.agent.cem_agent import CEMAgent


def test_cem():
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
