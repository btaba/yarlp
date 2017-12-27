"""
    Regression tests for baselines
"""

import pytest
import gym

from yarlp.agent.baseline_agents import RandomAgent

env = gym.make('CartPole-v0')


def test_random():
    agent = RandomAgent(env)
    agent.train(num_train_steps=1)
