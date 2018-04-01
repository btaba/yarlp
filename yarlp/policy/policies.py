"""
Defines policies
"""

import numpy as np
import tensorflow as tf
from yarlp.utils import tf_utils
from yarlp.policy.distributions import Categorical, DiagonalGaussian
from yarlp.model.networks import mlp
from yarlp.utils.env_utils import GymEnv


class Policy:

    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_outputs = GymEnv.get_env_action_space_dim(env)
        self._distribution = None
        self._scope = None

    @property
    def distribution(self):
        return self._distribution

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 self._scope.name)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self._scope.name)

    def predict(self, session, observations, greedy=False):
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 0)
        feed = {self.model['input:observations']: observations}

        if not greedy:
            return session.run(
                self.model['pi:sample_op'],
                feed)
        return session.run(
            self.model['pi:sample_greedy_op'],
            feed)


class CategoricalPolicy(Policy):

    def __init__(self, env, name, model, network_params,
                 input_node_name='observations',
                 action_name='action',
                 input_shape=None, network=mlp,
                 reuse=False, **kwargs):
        super().__init__(env)

        if input_shape is None:
            input_shape = [None] + list(self.observation_space.shape)

        if hasattr(env, 'is_atari') and env.is_atari:
            input_node = tf_utils.get_placeholder(
                name=input_node_name,
                dtype=tf.uint8, shape=input_shape)
            input_node = tf.cast(input_node, tf.float32) / 255.0
            model.add_input_node(input_node, input_node_name)
        else:
            input_node = tf_utils.get_placeholder(
                name=input_node_name,
                dtype=tf.float32, shape=input_shape)
            model.add_input_node(input_node, input_node_name)

        self.model = model

        model[action_name] = tf_utils.get_placeholder(
            name=action_name,
            dtype=tf.int32, shape=(None, 1))

        with tf.variable_scope(name, reuse=reuse) as s:
            self._scope = s
            output = network(inputs=input_node,
                             num_outputs=self.num_outputs,
                             **network_params)

            self._distribution = Categorical(model, output)


class GaussianPolicy(Policy):
    def __init__(self, env, name, model, network_params,
                 init_std=1.0, adaptive_std=False,
                 input_node_name='observations',
                 action_name='action',
                 input_shape=None, network=mlp,
                 reuse=False, **kwargs):
        super().__init__(env)
        self.adaptive_std = adaptive_std
        if input_shape is None:
            input_shape = [None] + list(self.observation_space.shape)

        input_node = tf_utils.get_placeholder(
            name=input_node_name,
            dtype=tf.float32, shape=input_shape)
        model.add_input_node(input_node, input_node_name)
        self.model = model

        model[action_name] = tf_utils.get_placeholder(
            name=action_name,
            dtype=tf.float32, shape=(None, self.num_outputs))

        with tf.variable_scope(name, reuse=reuse) as s:
            self._scope = s

            mean = network(inputs=input_node, num_outputs=self.num_outputs,
                           activation_fn=None,
                           **network_params)

            if adaptive_std:
                log_std = network(inputs=input_node,
                                  num_outputs=self.num_outputs,
                                  activation_fn=None,
                                  weights_initializer=tf.zeros_initializer(),
                                  **network_params)
            else:
                log_std = tf.get_variable(
                    name='logstd',
                    shape=[1, self.num_outputs],
                    initializer=tf.zeros_initializer())
                log_std = mean * 0.0 + log_std

            self._distribution = DiagonalGaussian(model, mean, log_std)

    def get_trainable_variables(self):
        tvars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            self._scope.name)
        if self.adaptive_std:
            return tvars
        return [t for t in tvars if not t.name.startswith('pi/logstd')]


def make_policy(env, name, model, **kwargs):
    if GymEnv.env_action_space_is_discrete(env):
        return CategoricalPolicy(
            env, name, model, **kwargs)
    return GaussianPolicy(
        env, name, model, **kwargs)
