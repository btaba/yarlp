import unittest
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.env_utils import NormalizedGymEnv


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
        env = NormalizedGymEnv(
            'CartPole-v0', normalize_obs=True, normalize_reward=True)
        env.reset()
        env.step(1)
        env.close()
        self.assertTrue(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 2)

    def test_continuous_action_space(self):
        env = NormalizedGymEnv(
            'MountainCarContinuous-v0', normalize_obs=True,
            normalize_reward=True)
        env.reset()
        env.step([0.1])
        env.close()
        self.assertFalse(GymEnv.env_action_space_is_discrete(env))
        self.assertEqual(GymEnv.get_env_action_space_dim(env), 1)
