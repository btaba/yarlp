# -*- coding: utf-8 -*-
"""
    Regression tests for the CEM agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.model.softmax_model import TFSoftmaxModel
from yarlp.agent.cem_agent import CEMAgent


class TestCEMCartPole(unittest.TestCase):
    """
    Cart-pole regression test on CEM agent
    """
    @classmethod
    def setUpClass(cls):
        env = gym.make('CartPole-v0')
        # cls.lm_keras = KerasSoftmaxModel(env)
        cls.lm_tf = TFSoftmaxModel(env)

    # def test_cem_keras(self):
    #     # TODO: consider taking an average reward over several agents
    #     # TODO: make sure that the trained policy returns an average reward that is high
    #     # instead of testing on the trained rewards
    #     agent = CEMAgent(
    #         self.lm_keras, num_training_steps=50, num_max_rollout_steps=1000,
    #         num_samples=10, init_var=.1, best_pct=0.2)
    #     r = agent.train(with_variance=True)

    #     # trained for 30 training steps
    #     self.assertEqual(len(r), 50)

    #     # last reward must be greater than first reward
    #     self.assertGreater(r[-1], r[0])

    #     # cart must have survived for more than 500 time steps
    #     self.assertTrue(r[-1] > 500)

    def test_cem_tf(self):
        # To solve the Cart-Pole we must get avg reward > 195
        # over 100 consecutive trials
        agent = CEMAgent(
            self.lm_tf, num_training_steps=50, num_max_rollout_steps=1000,
            num_samples=10, init_var=.1, best_pct=0.2)
        agent.train(with_variance=True)

        # # check that we trained for the number training steps specified
        # self.assertEqual(len(r), 50)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)
