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
        self.env = env
        self._distribution = None
        self._scope = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

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
        feed = {self.model['input:']: observations}

        if not greedy:
            return session.run(
                self.model['sample_op'],
                feed)
        return session.run(
            self.model['sample_greedy_op'],
            feed)


class CategoricalPolicy(Policy):

    def __init__(self, env, name, model, network_params,
                 input_shape=None, network=mlp):
        super().__init__(env)

        if input_shape is None:
            shape = [None] + list(self.observation_space.shape)
        num_outputs = GymEnv.get_env_action_space_dim(self.env)

        input_node = tf_utils.get_placeholder(
            name="observations",
            dtype=tf.float32, shape=shape)
        model.add_input_node(input_node)
        self.model = model

        model['action'] = tf.placeholder(
            dtype=tf.int32, shape=(None), name='action')

        with tf.variable_scope(name) as s:
            self._scope = s
            output = network(inputs=input_node,
                             num_outputs=num_outputs,
                             **network_params)

            self._distribution = Categorical(model, output)


class GaussianPolicy(Policy):
    def __init__(self, env, name, model, network_params, input_shape=None,
                 init_std=1.0, adaptive_std=False,
                 network=mlp):
        super().__init__(env)
        self.adaptive_std = adaptive_std
        if input_shape is None:
            shape = [None] + list(self.observation_space.shape)
        num_outputs = GymEnv.get_env_action_space_dim(self.env)

        input_node = tf_utils.get_placeholder(
            name='observations',
            dtype=tf.float32, shape=shape)
        model.add_input_node(input_node)
        self.model = model

        model['action'] = tf.placeholder(
            dtype=tf.float32, shape=(None, num_outputs), name='action')

        with tf.variable_scope(name) as s:
            self._scope = s
            mean = network(inputs=input_node, num_outputs=num_outputs,
                           activation_fn=None,
                           **network_params)

            if adaptive_std:
                log_std = network(inputs=input_node,
                                  num_outputs=num_outputs,
                                  activation_fn=None,
                                  weights_initializer=tf.zeros_initializer(),
                                  **network_params)
            else:
                log_std = tf.get_variable(
                    name='logstd',
                    shape=[1, num_outputs],
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


def make_policy(env, name, model, network_params={}, input_shape=None,
                init_std=1.0, adaptive_std=False, network=mlp):
    if GymEnv.env_action_space_is_discrete(env):
        return CategoricalPolicy(
            env, name, model, network_params=network_params,
            input_shape=input_shape, network=network)
    return GaussianPolicy(
        env, name, model, network_params=network_params,
        input_shape=input_shape, init_std=init_std,
        adaptive_std=adaptive_std, network=network)
