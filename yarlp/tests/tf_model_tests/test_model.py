"""
    Regression tests for the Model
"""

import gym
import shutil
import pytest
import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils
from functools import partial
from yarlp.model.model import Model


def build_graph(model):
    # create vars and input
    model.add_input('state')
    ini = tf.contrib.layers.variance_scaling_initializer()
    network = partial(
        tf.contrib.layers.fully_connected,
        weights_initializer=ini,
        activation_fn=None)
    model['output_node'] = model.add_output(
        network, num_outputs=1,
        input_node=model['input:state'])
    model['target_value'] = tf.placeholder(
        dtype=tf.float32, shape=(None,), name='target_value')

    # make a loss function and optimizer
    loss = tf.squared_difference(
        model['output_node'], model['target_value'])
    model.add_loss(loss)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=1)
    model.add_optimizer(optimizer, loss)

    # make gradient ops
    model.create_gradient_ops_for_node(optimizer, loss)

    model['vars'] = tf_utils.flatten_vars(tf.trainable_variables())


def build_update_feed_dict(model, state, target_value):
    feed_dict = {model['input:state']: np.expand_dims(state, 0),
                 model['target_value']: [target_value]}
    return feed_dict


def test_create_model():
    env = gym.make('CartPole-v0')
    M = Model(env, build_graph, build_update_feed_dict)
    assert M is not None


def test_update_model():
    env = gym.make('CartPole-v0')
    M = Model(env, build_graph, build_update_feed_dict)
    loss = M.update([0, 0, 0, 0], 2)
    assert loss[0] == [4.]


def test_gradient_ops():
    env = gym.make('CartPole-v0')
    M = Model(env, build_graph, build_update_feed_dict)
    feed_dict = M.build_update_feed([0, 1, 0, 1], 2)

    weights_before = M.get_weights()[0]
    M.apply_gradient_ops(M.get_loss().name, feed_dict)
    weights_after = M.get_weights()[0]

    assert weights_after[0] == pytest.approx(weights_before[0], 1e-7)
    assert weights_after[2] == pytest.approx(weights_before[2], 1e-7)
    assert weights_after[1] != weights_before[1]
    assert weights_after[3] != weights_before[3]


def test_load_and_save():
    env = gym.make('CartPole-v0')
    M = Model(env, build_graph, build_update_feed_dict)
    M.update([0, 0, 0, 0], 2)
    weights = M.G._session.run(M['vars'])
    M.save('test_load_and_save_model')
    M.update([0, 0, 0, 0], 2)
    del M
    M = Model.load(path='test_load_and_save_model')
    weights2 = M.G._session.run(M['vars'])
    M.build_update_feed([0, 1, 0, 1], 2)
    M.update([0, 0, 0, 0], 2)
    shutil.rmtree('test_load_and_save_model')
    assert np.allclose(weights, weights2)
