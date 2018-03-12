"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import pytest
import numpy as np
import shutil
from yarlp.utils.env_utils import ParallelEnvs
from yarlp.agent.a2c_agent import A2CAgent


env = ParallelEnvs('BeamRiderNoFrameskip-v4', 4)


def test_ddqn():
    agent = A2CAgent(env, max_timesteps=10, n_steps=2)
    agent.train()


def test_seed():
    agent = A2CAgent(env, seed=143, max_timesteps=10)
    agent.train()
    ob = agent.env.reset()

    agent = A2CAgent(env, seed=143, max_timesteps=10)
    agent.train()
    ob2 = agent.env.reset()

    assert np.all(
        np.array(ob) == np.array(ob2))


def test_save_models():
    agent = A2CAgent(env, max_timesteps=10)
    agent.train()
    agent.save('testy_a2c')
    agent = A2CAgent.load('testy_a2c')
    agent.t = 0
    agent.train()
    shutil.rmtree('testy_a2c')
