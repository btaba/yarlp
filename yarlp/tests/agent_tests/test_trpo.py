"""
    Regression tests for TRPO
"""

import pytest
import gym

from yarlp.agent.trpo_agent import TRPOAgent


def test_discrete_action_space():
    env = gym.make("CartPole-v1")
    agent = TRPOAgent(
        env, baseline_network=None,
        discount_factor=.95)
    agent.train(num_train_steps=1)


def test_continuous_action_space():
    env = gym.make("MountainCarContinuous-v0")
    agent = TRPOAgent(
        env, baseline_network=None,
        discount_factor=.95)
    agent.train(num_train_steps=1)
