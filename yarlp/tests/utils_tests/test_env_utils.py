import pytest
import numpy as np
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.utils.env_utils import RunningMeanStd
from yarlp.utils.env_utils import ParallelEnvs


def test_discrete_action_space():
    env = GymEnv('CartPole-v0')
    env.reset()
    env.step(1)
    env.close()
    assert GymEnv.env_action_space_is_discrete(env) is True
    assert GymEnv.get_env_action_space_dim(env) == 2


def test_continuous_action_space():
    env = GymEnv('MountainCarContinuous-v0')
    env.reset()
    env.step([0.1])
    env.close()
    assert GymEnv.env_action_space_is_discrete(env) is False
    assert GymEnv.get_env_action_space_dim(env) == 1


def test_discrete_action_space_norm():
    env = NormalizedGymEnv('CartPole-v0')
    env.reset()
    env.step(1)
    env.close()
    assert GymEnv.env_action_space_is_discrete(env) is True
    assert GymEnv.get_env_action_space_dim(env) == 2


def test_continuous_action_space_norm():
    env = NormalizedGymEnv('MountainCarContinuous-v0')
    env.reset()
    env.step([0.1])
    env.close()
    assert GymEnv.env_action_space_is_discrete(env) is False
    assert GymEnv.get_env_action_space_dim(env) == 1


def test_norm_obs():
    env = NormalizedGymEnv('MountainCarContinuous-v0', normalize_obs=True)
    env.reset()
    [env.step([0.01]) for _ in range(env.spec.timestep_limit + 1)]


def test_norm_reward():
    env = NormalizedGymEnv(
        'MountainCarContinuous-v0', normalize_rewards=True)
    env.reset()
    [env.step([0.01]) for _ in range(env.spec.timestep_limit + 1)]


def test_running_mean_std():
    rms = RunningMeanStd((2), clip_val=2)
    X = np.array([[1, 2], [2, 3], [2, 1], [5, 3]])
    for i in range(X.shape[0]):
        rms.cache(X[i])
        rms.update()
    assert np.allclose(rms._mean, X.mean(axis=0))
    assert np.allclose(rms._std, X.std(axis=0, ddof=1))


def test_longer_vec():
    np.random.seed(0)
    X = np.random.randn(100, 10)

    rms = RunningMeanStd(min_std=0.0, shape=(10,))

    for i in range(X.shape[0]):
        rms.cache(X[i])
        if i % 4 == 0:
            rms.update()
    rms.update()
    print(rms._mean, X.mean(axis=0))
    assert np.allclose(rms._mean, X.mean(axis=0))
    assert np.allclose(rms._std, X.std(axis=0, ddof=1))


def test_clip():
    rms = RunningMeanStd((1), clip_val=2)
    assert np.equal(rms.normalize(3), 2.)
    assert np.equal(rms.normalize(-3), -2.)


def test_parallel_env():
    env = ParallelEnvs('BreakoutNoFrameskip-v4', 3)
    assert env.reset().shape[0] == 3
    assert env.step([0, 0, 0])[0].shape[0] == 3
    assert np.all(np.array(env.get_total_steps()) > 1)
