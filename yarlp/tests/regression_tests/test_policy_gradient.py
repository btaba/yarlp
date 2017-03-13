"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.model.model_factories import policy_gradient_model_factory
from yarlp.agent.policy_gradient_agents import REINFORCEAgent
from yarlp.agent.policy_gradient_agents import OneStepActorCriticPG


class TestREINFORCECartPole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        env = gym.make('CartPole-v0')
        cls.pm = policy_gradient_model_factory(env)

    def test_reinforce_discrete(self):
        # To solve the Cart-Pole we must get avg reward > 195
        # over 100 consecutive trials
        agent = REINFORCEAgent(
            self.pm, num_max_rollout_steps=1000,
            discount_factor=.95)
        agent.train(num_training_steps=100, with_baseline=False)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)

    def test_reinforce_w_linear_baseline(self):
        agent = REINFORCEAgent(
            self.pm, value_model='linear', num_max_rollout_steps=1000,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)

    def test_reinforce_w_average_baseline(self):
        agent = REINFORCEAgent(
            self.pm, value_model='average', num_max_rollout_steps=1000,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(sampled_greedy_rewards) > 195)


class TestActorCriticPG(unittest.TestCase):

    def test_continuous(self):
        # One-step PG doesn't work well with CartPole, maybe TD(lambda) will
        env = gym.make('MountainCarContinuous-v0')
        pm = policy_gradient_model_factory(
            env, action_space='continuous', learning_rate=0.01)
        agent = OneStepActorCriticPG(
            pm, num_max_rollout_steps=10000,
            discount_factor=.95)
        agent.train(num_training_steps=60)

        sampled_greedy_rewards = []
        for i in range(100):
            sampled_greedy_rewards.append(
                agent.do_greedy_episode(max_time_steps=10000))

        print(sampled_greedy_rewards)
        print(np.mean(sampled_greedy_rewards))
        self.assertTrue(np.mean(sampled_greedy_rewards) >= -110)
