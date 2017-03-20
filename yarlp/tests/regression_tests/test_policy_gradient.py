"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np

from yarlp.agent.policy_gradient_agents import REINFORCEAgent
from yarlp.agent.policy_gradient_agents import ActorCriticPG


class TestREINFORCECartPole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('CartPole-v0')

    def test_reinforce_no_baseline(self):
        # To solve the Cart-Pole we must get avg reward > 195
        # over 100 consecutive trials
        agent = REINFORCEAgent(
            self.env,
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=100, with_baseline=False)

        sampled_greedy_rewards = []
        for i in range(self.env.spec.trials):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(
            sampled_greedy_rewards) > self.env.spec.reward_threshold)

    def test_reinforce_w_linear_baseline(self):
        agent = REINFORCEAgent(
            self.env, value_model='linear',
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(self.env.spec.trials):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(
            sampled_greedy_rewards) > self.env.spec.reward_threshold)

    def test_reinforce_w_average_baseline(self):
        agent = REINFORCEAgent(
            self.env, value_model='average',
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(self.env.spec.trials):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(
            sampled_greedy_rewards) > self.env.spec.reward_threshold)


class TestActorCriticPG(unittest.TestCase):

    def test_continuous(self):
        env = gym.make('MountainCarContinuous-v0')
        agent = ActorCriticPG(
            env, action_space='continuous',
            policy_learning_rate=0.1,
            num_max_rollout_steps=8000,
            discount_factor=1, lambda_p=0.5,
            lambda_v=1,
            value_model_learning_rate=1)
        agent.train(num_training_steps=1)

        sampled_greedy_rewards = []
        for i in range(env.spec.trials):
            sampled_greedy_rewards.append(
                agent.do_greedy_episode(
                    max_time_steps=8000))

        print(sampled_greedy_rewards)
        print(np.mean(sampled_greedy_rewards))
        self.assertTrue(np.mean(
            sampled_greedy_rewards) > env.spec.reward_threshold)

#     def test_discrete(self):
#         # Cartpole doesn't work well with actor critic
#         env = gym.make('CartPole-v0')
#         pm = policy_gradient_model_factory(
#             env, action_space='discrete', learning_rate=.1)
#         agent = ActorCriticPG(
#             pm, num_max_rollout_steps=1000,
#             discount_factor=1, lambda_p=1,
#             lambda_v=1,
#             value_model_learning_rate=.1)
#         agent.train(num_training_steps=300)

#         sampled_greedy_rewards = []
#         for i in range(100):
#             sampled_greedy_rewards.append(
#                 agent.do_greedy_episode(max_time_steps=1000))

#         print(sampled_greedy_rewards)
#         print(np.mean(sampled_greedy_rewards))
#         self.assertTrue(np.mean(sampled_greedy_rewards) > 195)
