import numpy as np
import tensorflow as tf

from yarlp.model.model import Model
from functools import partial


def value_function_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01, input_shape=None, model_file_path=None):
    """Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr, shape):
        input_node = model.add_input(shape=shape)

        # vsi = tf.contrib.layers.variance_scaling_initializer()
        # network = partial(network, activation_fn=None,
        #                   weights_initializer=vsi)
        network = partial(network, activation_fn=None)
        output_node = model.add_output(network, num_outputs=1)
        model.value = output_node

        # Value function estimation stuff
        model.state = input_node
        model.target_value = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        loss = tf.reduce_mean(
            tf.squared_difference(output_node, model.target_value))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(loss)
        model.add_optimizer(optimizer, loss)
        model.learning_rate = lr

        model.create_gradient_ops_for_node(optimizer, output_node)

    def build_update_feed_dict(model, state, target_value):
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
        feed_dict = {model.state: state,
                     model.target_value: target_value}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, shape=input_shape)

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


def discrete_pg_model_factory(
        env, network, learning_rate=0.01,
        entropy_weight=0.001,
        input_shape=None, model_file_path=None):
    """
    Policy model for discrete action spaces with policy gradient update
    """
    def build_graph(model, network, lr, input_shape):
        input_node = model.add_input(shape=input_shape)

        model.state = input_node
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.learning_rate = lr

        # Softmax policy for discrete action spaces
        network = partial(network, activation_fn=tf.nn.softmax)
        model.output_node = model.add_output(network)
        model.action = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='action')
        action_one_hot = tf.one_hot(model.action, model.output_node.shape[1])

        model.pi = tf.squeeze(tf.reduce_sum(
            action_one_hot * model.output_node, 1))
        model.log_pi = tf.squeeze(tf.log(model.pi + 1e-8))
        # model.pi = tf.reduce_sum(
        #     action_one_hot * model.output_node, 1)
        # model.log_pi = tf.log(model.pi + 1e-8)

        model.loss = -tf.reduce_mean(
            model.log_pi * model.Return) +\
            entropy_weight * tf.reduce_mean(model.log_pi * model.pi)
        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(model.loss)
        model.add_optimizer(model.optimizer, model.loss)

        model.create_gradient_ops_for_node(model.optimizer, model.log_pi)

    def build_update_feed_dict(model, state, return_, action):
        feed_dict = {model.state: state,
                     model.Return: np.squeeze([return_]), model.action: action}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, input_shape=input_shape)

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


def continuous_gaussian_pg_model_factory(
        env, network, learning_rate=0.01, entropy_weight=0.001,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        input_shape=None, model_file_path=None):
    """
    Policy model for continuous action spaces with policy gradient update
    """
    def build_graph(model, network, lr, input_shape):
        input_node = model.add_input(shape=input_shape)

        model.state = input_node
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.learning_rate = lr

        network = partial(network, activation_fn=None)
        model.mean = model.add_output(network, name='greedy')
        model.mean = tf.squeeze(model.mean)

        if adaptive_std:
            model.log_std = model.add_output(network, name='log_std')
        else:
            model.log_std = tf.log(init_std)
        model.std = tf.maximum(tf.exp(model.log_std), min_std)
        model.std = tf.squeeze(model.std)

        model.normal_dist = tf.contrib.distributions.Normal(
            model.mean, model.std)

        model.output = tf.squeeze(model.normal_dist.sample([1]))

        # model.action = tf.clip_by_value(
        #     model.action, model._env.action_space.low[0],
        #     model._env.action_space.high[0])

        model.add_output_node(model.output)

        model.action = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='action')

        model.log_pi = tf.squeeze(model.normal_dist.log_prob(model.action))
        model.pi = tf.squeeze(model.normal_dist.prob(model.action))
        model.loss = -tf.reduce_mean(model.log_pi * model.Return) +\
            entropy_weight * tf.reduce_mean(model.log_pi * model.pi)

        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

        model.add_loss(model.loss)
        model.add_optimizer(model.optimizer, model.loss)

    def build_update_feed_dict(model, state, return_, action):
        feed_dict = {model.state: state,
                     model.Return: np.squeeze([return_]), model.action: action}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, input_shape=input_shape)

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


# def ddpg_actor_critic_model_factory(
#         env, actor_network, critic_network, actor_learning_rate=0.01,
#         critic_learning_rate=0.01, input_shape=None, batch_norm=False):
#     """Actor and Critic models for deterministic policy gradient
#     """
#     def build_graph(model, actor_network, critic_network,
#                     actor_learning_rate, critic_learning_rate,
#                     input_shape, batch_norm):
#         model.state = model.add_input(shape=input_shape, name='')

#         # TODO:
#         # Add batch_norm
#         # Add the action at a different layer instead of the first layer

#         # if batch_norm:
#         #     tf.contrib.layers.batch_norm()

#         # Actor
#         with tf.variable_scope('actor'):
#             a_network = partial(actor_network, activation_fn=None)
#             model.action = model.add_output(a_network, name='')

#         # Critic
#         model.action_input = model.add_input(
#             shape=model.action.get_shape(), name='action')
#         q_input = tf.concat([model.action_input, model.state], axis=1)
#         with tf.variable_scope('critic'):
#             c_network = partial(
#                 critic_network, activation_fn=None)
#             model.q_value = model.add_output(
#                 c_network, input_node=q_input, name='critic')

#         # Critic optimizer
#         model.td_value = tf.placeholder(
#             dtype=tf.float32, shape=(None,), name='td_value')
#         model.critic_loss = tf.reduce_mean(tf.squared_difference(
#             model.q_value, model.td_value))
#         model.critic_optimizer = tf.train.AdamOptimizer(
#             learning_rate=critic_learning_rate)
#         # with tf.control_dependencies(tf.GraphKeys.UPDATE_OPS):
#         model.critic_optimizer_op = model.critic_optimizer.minimize(
#             model.critic_loss)

#         # Actor optimizer
#         q_input2 = tf.concat([model.action, model.state], axis=1)
#         with tf.variable_scope("critic", reuse=True):
#             model.q_value_w_actor = model.add_output(
#                 c_network, input_node=q_input2,
#                 name='critic_with_actor_input')

#         model.actor_loss = tf.reduce_mean(model.q_value_w_actor)
#         model.actor_optimizer = tf.train.AdamOptimizer(
#             learning_rate=actor_learning_rate)
#         actor_vars = model.G.trainable_variables_for_scope('actor')
#         # with tf.control_dependencies(tf.GraphKeys.UPDATE_OPS):
#         model.actor_optimizer_op = model.actor_optimizer.minimize(
#             model.actor_loss, var_list=actor_vars)

#     def build_update_feed_dict(model, state, action):
#         pass

#     build_graph = partial(build_graph, actor_network=actor_network,
#                           critic_network=critic_network,
#                           actor_learning_rate=actor_learning_rate,
#                           critic_learning_rate=critic_learning_rate,
#                           input_shape=input_shape, batch_norm=batch_norm)

#     return Model(env, build_graph, build_update_feed_dict)
