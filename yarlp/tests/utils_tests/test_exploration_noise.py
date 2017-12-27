import gym
import pytest
from yarlp.utils.exploration_noise import AnnealedGaussian, OrnsteinUhlenbeck


def test_annealed_gaussian():
    ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
    s = ag.sample_noise()
    assert s.shape == (1,)


def test_annealed_gaussian_annealing():
    ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
    assert ag.sigma == pytest.approx(1)

    ag.sample_noise()
    assert ag.sigma == pytest.approx(1 - (1 - .1) * 1 / 3.0)

    ag.sample_noise()
    assert ag.sigma == pytest.approx(1 - (1 - .1) * 2 / 3.0)

    ag.sample_noise()
    assert ag.sigma == pytest.approx(1 - (1 - .1) * 3 / 3.0)

    for _ in range(100):
        ag.sample_noise()
    assert ag.sigma == pytest.approx(.1)


def test_add_noise_to_action():
    env = gym.make('MountainCarContinuous-v0')
    ag = AnnealedGaussian(0, 1, .1, n_actions=1, n_annealing_steps=3)
    ag.add_noise_to_action(env, 1)


def test_ou():
    ou = OrnsteinUhlenbeck(0, .15, .3, n_actions=1)
    s = ou.sample_noise()
    assert len(s) == 1

    ou.reset()
    assert ou._x[0] == 0
