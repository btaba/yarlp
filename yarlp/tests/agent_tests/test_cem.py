"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import pytest
import gym
import shutil

from yarlp.agent.cem_agent import CEMAgent


def test_cem():
    env = gym.make('CartPole-v0')
    agent = CEMAgent(
        env,
        n_weight_samples=25, init_var=.1, best_pct=0.2)
    agent.train(num_train_steps=1, with_variance=True)


def test_cem_save_model():
    env = gym.make('CartPole-v0')
    agent = CEMAgent(
        env,
        n_weight_samples=25, init_var=.1, best_pct=0.2)
    agent.save('testy_cem')

    agent = CEMAgent.load('testy_cem')
    agent.train(num_train_steps=1)

    shutil.rmtree('testy_cem')
