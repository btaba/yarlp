"""
Defines policies
"""

import tensorflow as tf
from yarlp.policies.distributions import Categorical, DiagonalGaussian
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


class CategoricalPolicy(Policy):

    def __init__(self, env, name, network_params,
                 input_shape=None, network=mlp):
        super().__init__(env)

        if input_shape is None:
            shape = (None, self.observation_space.shape[0])
        num_outputs = GymEnv.get_env_action_space_dim(self.env)

        with tf.variable_scope(name):
            self._scope = tf.get_variable_scope()
            self.input_node = tf.placeholder(name="observations",
                                             dtype=tf.float32, shape=shape)
            self.output = network(inputs=self.input_node,
                                  num_outputs=num_outputs,
                                  activation_fn=tf.nn.softmax,
                                  **network_params)

            self.action_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='action')

            self._distribution = Categorical(self.output)


class GaussianPolicy(Policy):
    def __init__(self, env, name, network_params, input_shape=None,
                 init_std=1.0, adaptive_std=False,
                 network=mlp):
        super().__init__(env)

        if input_shape is None:
            shape = (None, self.observation_space.shape[0])
        num_outputs = GymEnv.get_env_action_space_dim(self.env)

        with tf.variable_scope(name):
            self._scope = tf.get_variable_scope()
            self.input_node = tf.placeholder(name="observations",
                                             dtype=tf.float32, shape=shape)
            mean = network(inputs=self.input_node, num_outputs=num_outputs,
                           activation_fn=None,
                           **network_params)

            if adaptive_std:
                log_std = mlp(inputs=self.input_node, num_outputs=num_outputs,
                              activation_fn=None,
                              weights_initializer=tf.zeros_initializer(),
                              **network_params)
            else:
                log_std = tf.log(tf.ones(shape=[1, num_outputs]) * init_std)

            self.action_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None,), name='action')

            self._distribution = DiagonalGaussian(mean, log_std)


def make_policy(env, name, network_params={}, input_shape=None,
                init_std=1.0, adaptive_std=False, network=mlp):
    if GymEnv.env_action_space_is_discrete(env):
        return CategoricalPolicy(
            env, name, network_params=network_params,
            input_shape=input_shape, network=network)
    return GaussianPolicy(
        env, name, network_params=network_params,
        input_shape=input_shape, init_std=init_std,
        adaptive_std=adaptive_std, network=network)
