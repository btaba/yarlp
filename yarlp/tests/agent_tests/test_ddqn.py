"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import pytest
import gym
import numpy as np
import shutil

from yarlp.agent.ddqn_agent import DDQNAgent


env = gym.make('PongNoFrameskip-v4')


def test_ddqn():
    agent = DDQNAgent(env, max_timesteps=2)
    agent.train()


def test_seed():
    agent = DDQNAgent(env, seed=143, max_timesteps=2)
    agent.train()
    ob, *_ = agent.replay_buffer.sample(5, 0.4)

    agent = DDQNAgent(env, seed=143, max_timesteps=2)
    agent.train()
    ob2, *_ = agent.replay_buffer.sample(5, 0.4)

    assert np.all(
        np.array(ob) == np.array(ob2))


def test_save_models():
    agent = DDQNAgent(env, max_timesteps=2)
    agent.train()
    agent.save('testy_ddqn')
    agent = DDQNAgent.load('testy_ddqn')
    agent.t = 0
    agent.train()
    shutil.rmtree('testy_ddqn')
