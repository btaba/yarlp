import numpy as np
import tensorflow as tf

from yarlp.policy.policies import make_policy
from yarlp.model.model import Model
from functools import partial
from yarlp.utils import tf_utils


def value_function_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01, input_shape=None, model_file_path=None):
    """
    Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr, shape):
        input_node = model.add_input(shape=shape)

        network = partial(network, activation_fn=None)
        output_node = model.add_output(network, num_outputs=1)
        model.value = output_node

        # Value function estimation stuff
        model.state = input_node
        model.target_value = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        model.loss = loss = tf.reduce_mean(
            tf.square(tf.squeeze(model.value) - tf.squeeze(model.target_value)))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(loss)
        model.add_optimizer(optimizer, loss)
        model.learning_rate = lr

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


def cem_model_factory(
        env, network, network_params={},
        input_shape=None,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        model_file_path=None):
    """
    Model for gradient method
    """

    def build_graph(model, network=network,
                    input_shape=input_shape,
                    network_params=network_params,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model.policy = policy
        model.output_node = policy.distribution.output_node
        model.add_output_node(model.output_node)

        var_list = policy.get_trainable_variables()
        shapes = map(tf_utils.var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        model.theta = tf.placeholder(tf.float32, [total_size])

        var_list = policy.get_trainable_variables()
        model.gf = tf_utils.flatten_vars(var_list)
        model.sff = tf_utils.setfromflat(var_list, model.theta)

    def build_update_feed_dict(model):
        pass

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


def pg_model_factory(
        env, network, network_params={}, learning_rate=0.01,
        entropy_weight=0.001, input_shape=None,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        model_file_path=None):
    """
    Model for gradient method
    """

    def build_graph(model, network=network, lr=learning_rate,
                    input_shape=input_shape,
                    network_params=network_params,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model.policy = policy
        model.state = model.add_input_node(policy.input_node)
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.output_node = policy.distribution.output_node
        model.add_output_node(model.output_node)

        model.action = policy.action_placeholder

        model.log_pi = policy.distribution.log_likelihood(model.action)
        entropy = tf.reduce_mean(policy.distribution.entropy())

        model.loss = -tf.reduce_mean(
            model.log_pi * model.Return) +\
            entropy_weight * entropy

        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(model.loss)
        model.add_optimizer(
            model.optimizer, model.loss,
            var_list=policy.get_trainable_variables())

    def build_update_feed_dict(model, state, return_, action):
        if len(action.shape) == 1:
            action = action.reshape(-1, 1)
        feed_dict = {model.state: state,
                     model.Return: np.squeeze([return_]), model.action: action}
        return feed_dict

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


def trpo_model_factory(
        env, network, network_params={},
        entropy_weight=0,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        input_shape=None, model_file_path=None):
    """
    Policy model for discrete action spaces with policy gradient update
    """
    def build_graph(model, network, input_shape):

        policy = make_policy(
            env, 'pi', network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model.policy = policy
        old_policy = make_policy(
            env, 'oldpi', network_params=network_params,
            input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model.old_policy = old_policy

        model.state = model.add_input_node(policy.input_node)
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.output_node = policy.distribution.output_node
        model.add_output_node(model.output_node)

        if hasattr(policy, 'mean'):
            model.add_output_node(policy.mean, name='greedy')

        model.action = policy.action_placeholder

        entropy = tf.reduce_mean(policy.distribution.entropy())
        model.kl = tf.reduce_mean(
            old_policy.distribution.kl(policy._distribution))
        entbonus = entropy_weight * entropy

        ratio = policy.distribution.likelihood_ratio(
            model.action, old_policy.distribution)
        model.surrgain = tf.reduce_mean(
            ratio * model.Return)

        model.optimgain = model.surrgain + entbonus
        model.losses = [model.optimgain, model.kl,
                        entbonus, model.surrgain, entropy]

        var_list = policy.get_trainable_variables()
        model.klgrads = tf.gradients(model.kl, var_list)

        model.logstd = policy.get_trainable_variables()[-1]

        model.pg = tf_utils.flatgrad(model.optimgain, var_list)

        shapes = map(tf_utils.var_shape, var_list)
        start = 0
        tangents = []
        model.flat_tangent = tf.placeholder(
            dtype=tf.float32, shape=[None], name='flat_tangent')
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(model.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size

        model.theta = tf.placeholder(tf.float32, [start])

        model.gvp = tf.add_n(
            [tf.reduce_sum(g * t) for (g, t) in zip(model.klgrads, tangents)])
        model.fvp = tf_utils.flatgrad(model.gvp, var_list)
        model.gf = tf_utils.flatten_vars(var_list)
        model.sff = tf_utils.setfromflat(var_list, model.theta)
        model.set_old_pi_eq_new_pi = [
            tf.assign(old, new) for (old, new) in
            zip(old_policy.get_variables(), policy.get_variables())]

    def build_update_feed_dict(model, state, return_, action):
        if len(action.shape) == 1:
            action = action.reshape(-1, 1)
        feed_dict = {model.state: state, model.old_policy.input_node: state,
                     model.Return: np.squeeze([return_]), model.action: action}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          input_shape=input_shape)

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)
