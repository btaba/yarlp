import gym
import unittest
from yarlp.utils.exploration_noise import AnnealedGaussian, OrnsteinUhlenbeck


class TestGaussianNoise(unittest.TestCase):

    def test_annealed_gaussian(self):
        ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
        s = ag.sample_noise()
        self.assertEqual(s.shape, (1,))

    def test_annealed_gaussian_annealing(self):
        ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
        self.assertAlmostEqual(ag.sigma, 1)

        ag.sample_noise()
        self.assertAlmostEqual(ag.sigma, 1 - (1 - .1) * 1 / 3.0)

        ag.sample_noise()
        self.assertAlmostEqual(ag.sigma, 1 - (1 - .1) * 2 / 3.0)

        ag.sample_noise()
        self.assertAlmostEqual(ag.sigma, 1 - (1 - .1) * 3 / 3.0)

        for _ in range(100):
            ag.sample_noise()
        self.assertAlmostEqual(ag.sigma, .1)

    def test_add_noise_to_action(self):
        env = gym.make('MountainCarContinuous-v0')
        ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
        ag.add_noise_to_action(env, 1)


class TestOUNoise(unittest.TestCase):

    def test_ou(self):
        ou = OrnsteinUhlenbeck(0, .15, .3, n_actions=1)
        s = ou.sample_noise()
        self.assertEqual(len(s), 1)

        ou.reset()
        self.assertEqual(ou._x[0], 0)
