"""
    Regression tests for the Model
"""

import gym
import unittest
import numpy as np
import tensorflow as tf
from functools import partial
from yarlp.model.model import Model


class testModel(unittest.TestCase):

    @staticmethod
    def build_graph(model):
        # create vars and input
        model.add_input('state')
        network = partial(
            tf.contrib.layers.fully_connected,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
            activation_fn=None)
        model.output_node = model.add_output(network, num_outputs=1)
        model['target_value'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')

        # make a loss function and optimizer
        loss = tf.squared_difference(
            model.output_node, model['target_value'])
        model.add_loss(loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=1)
        model.add_optimizer(optimizer, loss)

        # make gradient ops
        model.create_gradient_ops_for_node(optimizer, loss)

    @staticmethod
    def build_update_feed_dict(model, state, target_value):
        feed_dict = {model['input:state']: np.expand_dims(state, 0),
                     model['target_value']: [target_value]}
        return feed_dict

    def test_create_model(self):
        env = gym.make('CartPole-v0')
        M = Model(env, testModel.build_graph, testModel.build_update_feed_dict)
        self.assertIsNotNone(M)

    def test_update_model(self):
        env = gym.make('CartPole-v0')
        M = Model(env, testModel.build_graph, testModel.build_update_feed_dict)
        loss = M.update([0, 0, 0, 0], 2)
        self.assertEqual(loss[0], [4.])

    def test_gradient_ops(self):
        env = gym.make('CartPole-v0')
        M = Model(env, testModel.build_graph, testModel.build_update_feed_dict)
        feed_dict = M.build_update_feed([0, 1, 0, 1], 2)

        weights_before = M.get_weights()[0]
        M.apply_gradient_ops(M.get_loss().name, feed_dict)
        weights_after = M.get_weights()[0]

        self.assertAlmostEqual(weights_after[0], weights_before[0], places=7)
        self.assertAlmostEqual(weights_after[2], weights_before[2], places=7)
        self.assertNotEqual(weights_after[1], weights_before[1])
        self.assertNotEqual(weights_after[3], weights_before[3])

    def test_load_and_save(self):
        env = gym.make('CartPole-v0')
        M = Model(env, testModel.build_graph, testModel.build_update_feed_dict)
        M.update([0, 0, 0, 0], 2)
        M.save('test_load_and_save_model')
        M = Model(env, None, testModel.build_update_feed_dict,
                  'test_load_and_save_model')
        M.build_update_feed([0, 1, 0, 1], 2)
        M.update([0, 0, 0, 0], 2)
