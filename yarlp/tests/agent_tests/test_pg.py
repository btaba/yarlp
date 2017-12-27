"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import pytest
import gym
import numpy as np
# import shutil

from yarlp.agent.pg_agents import REINFORCEAgent
from yarlp.agent.base_agent import do_rollout


env = gym.make('CartPole-v0')


def test_reinforce_no_baseline():
    agent = REINFORCEAgent(
        env, baseline_network=None,
        discount_factor=.95)
    agent.train(num_train_steps=1)


def test_reinforce_w_baseline():
    agent = REINFORCEAgent(
        env,
        discount_factor=.95)
    agent.train(num_train_steps=1)


def test_seed():
    agent = REINFORCEAgent(env, seed=143)
    r = next(do_rollout(agent, env, n_steps=2))

    agent = REINFORCEAgent(env, seed=143)
    r2 = next(do_rollout(agent, env, n_steps=2))
    assert np.all(
        np.array(r['actions']) == np.array(r2['actions']))
    assert np.all(
        np.array(r['observations']) == np.array(r2['observations']))
    assert np.all(
        np.array(r['rewards']) == np.array(r2['rewards']))

# def test_reinforce_save_models():
#     agent = REINFORCEAgent(
#         env,
#         discount_factor=.95)
#     agent.save_models('testy_reinforce')
#     agent = REINFORCEAgent(
#         env,
#         model_file_path='testy_reinforce')
#     shutil.rmtree('testy_reinforce')

# def tearDown():
#     shutil.rmtree()
