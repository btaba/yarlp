"""
    Regression tests for the REINFORCE agent on OpenAI gym environments
"""

import unittest
import gym
import numpy as np
import tensorflow as tf

from yarlp.agent.dpg_agents import DDPG


class TestActorCriticPG(unittest.TestCase):

    def test_ddpg(self):
        vsi = tf.contrib.layers.variance_scaling_initializer()

        def network1(inputs, num_outputs, activation_fn,
                     weights_initializer=None, **kwargs):
            x = tf.contrib.layers.fully_connected(
                inputs, num_outputs=32,
                weights_initializer=weights_initializer,
                activation_fn=tf.nn.relu, **kwargs)
            x = tf.contrib.layers.fully_connected(
                x, num_outputs=32,
                weights_initializer=weights_initializer,
                activation_fn=tf.nn.relu, **kwargs)
            x = tf.contrib.layers.fully_connected(
                x, num_outputs=num_outputs,
                activation_fn=activation_fn,
                weights_initializer=weights_initializer, **kwargs)
            return x

        env = gym.make('MountainCarContinuous-v0')
        agent = DDPG(env, num_max_rollout_steps=env.spec.timestep_limit,
                     actor_network=network1, critic_network=network1)
        agent.train(num_training_steps=300)

        sampled_rewards = []
        for i in range(env.spec.trials):
            sampled_rewards.append(
                len(agent.rollout().rewards))

        print(sampled_rewards)
        print(np.mean(sampled_rewards))
        self.assertTrue(np.mean(
            sampled_rewards) < 5000)
