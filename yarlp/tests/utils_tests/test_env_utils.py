import unittest
import numpy as np
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.utils.env_utils import RunningMeanStd


class TestGymEnv(unittest.TestCase):

    def test_discrete_action_space(self):
        env = GymEnv('CartPole-v0')
        env.reset()
        env.step(1)
        env.close()
        self.assertTrue(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 2)

    def test_continuous_action_space(self):
        env = GymEnv('MountainCarContinuous-v0')
        env.reset()
        env.step([0.1])
        env.close()
        self.assertFalse(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 1)


class TestNormalizedGymEnv(unittest.TestCase):

    def test_discrete_action_space(self):
        env = NormalizedGymEnv('CartPole-v0')
        env.reset()
        env.step(1)
        env.close()
        self.assertTrue(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 2)

    def test_continuous_action_space(self):
        env = NormalizedGymEnv('MountainCarContinuous-v0')
        env.reset()
        env.step([0.1])
        env.close()
        self.assertFalse(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 1)

    def test_norm_obs(self):
        env = NormalizedGymEnv('MountainCarContinuous-v0', normalize_obs=True)
        env.reset()
        [env.step([0.01]) for _ in range(env.spec.timestep_limit + 1)]

    def test_norm_reward(self):
        env = NormalizedGymEnv(
            'MountainCarContinuous-v0', normalize_rewards=True)
        env.reset()
        [env.step([0.01]) for _ in range(env.spec.timestep_limit + 1)]


class TestRunningMeanStd(unittest.TestCase):

    def test_running_mean_std(self):
        rms = RunningMeanStd((2), clip_val=2)
        X = np.array([[1, 2], [2, 3], [2, 1], [5, 3]])
        for i in range(X.shape[0]):
            rms.cache(X[i])
            rms.update()
        assert np.allclose(rms._mean, X.mean(axis=0))
        assert np.allclose(rms._std, X.std(axis=0, ddof=1))

    def test_clip(self):
        rms = RunningMeanStd((1), clip_val=2)
        assert np.equal(rms.normalize(3), 2.)
        assert np.equal(rms.normalize(-3), -2.)
