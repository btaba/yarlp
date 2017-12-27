"""
Test the base agent abstract class by instantiating a RandomAgent
"""

import pytest
import gym
import numpy as np


from yarlp.agent.baseline_agents import RandomAgent
from yarlp.agent.base_agent import do_rollout


env = gym.make('CartPole-v0')


def test_rollout():
    agent = RandomAgent(env)
    r = do_rollout(agent, env)
    r.__next__()


def test_rollout_n_steps():
    agent = RandomAgent(env)
    r = do_rollout(agent, env, n_steps=2)
    r.__next__()


def test_seed():
    agent = RandomAgent(env, seed=143)
    r = next(do_rollout(agent, env, n_steps=2))

    agent = RandomAgent(env, seed=143)
    r2 = next(do_rollout(agent, env, n_steps=2))
    assert np.all(
        np.array(r['actions']) == np.array(r2['actions']))
    assert np.all(
        np.array(r['observations']) == np.array(r2['observations']))
    assert np.all(
        np.array(r['rewards']) == np.array(r2['rewards']))
