"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import pytest
import numpy as np
import shutil
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.agent.ddqn_agent import DDQNAgent


env = NormalizedGymEnv(
    'PongNoFrameskip-v4',
    is_atari=True
)


def test_ddqn():
    agent = DDQNAgent(env, max_timesteps=10,
                      learning_start_timestep=1,
                      train_freq=5,
                      batch_size=1)
    agent.train()


def test_seed():
    agent = DDQNAgent(env, seed=143, max_timesteps=2)
    agent.train()
    ob, *_ = agent.replay_buffer.sample(1)

    agent = DDQNAgent(env, seed=143, max_timesteps=2)
    agent.train()
    ob2, *_ = agent.replay_buffer.sample(1)

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
