"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np
import tensorflow as tf

from yarlp.agent.pg_agents import REINFORCEAgent
# from yarlp.agent.pg_agents import ActorCriticPG


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

    def test_reinforce_w_deep_baseline(self):
        agent = REINFORCEAgent(
            self.env,
            baseline_network=tf.contrib.layers.fully_connected,
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
            self.env,
            num_max_rollout_steps=self.env.spec.timestep_limit,
            discount_factor=.95)
        agent.train(num_training_steps=100)

        sampled_greedy_rewards = []
        for i in range(self.env.spec.trials):
            sampled_greedy_rewards.append(agent.do_greedy_episode())

        print(sampled_greedy_rewards)
        self.assertTrue(np.mean(
            sampled_greedy_rewards) > self.env.spec.reward_threshold)


# class TestActorCriticPG(unittest.TestCase):

#     def test_continuous_with_traces(self):
#         # this algorithm is not working well

#         env = gym.make('MountainCarContinuous-v0')

#         def state_featurizer(state, high_state, low_state,
#                              n_tiles_per_dim=200):
#             # super simple tile coder
#             state = np.copy(state)
#             new_state = np.zeros((n_tiles_per_dim * state.shape[0]))
#             for s in zip(state, high_state, low_state):
#                 s = (s[0] - s[2]) / (s[1] - s[2])
#                 s *= (n_tiles_per_dim - 1)
#                 new_state[int(s) % n_tiles_per_dim] = 1
#             return new_state

#         from functools import partial
#         state_featurizer = partial(
#             state_featurizer, high_state=env.observation_space.high,
#             low_state=env.observation_space.low)

#         agent = ActorCriticPG(
#             env, action_space='continuous',
#             policy_learning_rate=0.001,
#             num_max_rollout_steps=5000,
#             discount_factor=0.995, lambda_p=0,
#             lambda_v=0, state_featurizer=state_featurizer,
#             value_model_learning_rate=0.5)
#         agent.train(num_training_steps=30)

#         sampled_rewards = []
#         for i in range(env.spec.trials):
#             sampled_rewards.append(
#                 len(agent.rollout().rewards))

#         print(sampled_rewards)
#         print(np.mean(sampled_rewards))
#         self.assertTrue(np.mean(
#             sampled_rewards) < 5000)
