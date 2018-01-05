import numpy as np
import tensorflow as tf

from yarlp.policy.policies import make_policy
from yarlp.model.model import Model
from functools import partial
from yarlp.model.networks import mlp
from yarlp.utils import tf_utils


def build_vf_update_feed_dict(model, state, target_value):
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)
    feed_dict = {model['state']: state,
                 model['target_value']: target_value}
    return feed_dict


def empty_feed_dict(*args):
    pass


def build_pg_update_feed_dict(model, state, return_, action):
    if len(action.shape) == 1:
        action = action.reshape(-1, 1)
    feed_dict = {model['state']: state,
                 model['Return']: np.squeeze([return_]),
                 model['action']: action}
    return feed_dict


def value_function_model_factory(
        env, network=mlp,
        learning_rate=0.01, input_shape=None, model_file_path=None,
        name='value_function'):
    """
    Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr, shape):
        input_node = model.add_input(shape=shape)

        network = partial(network, final_activation_fn=None)
        output_node = model.add_output(network, num_outputs=1)
        model['value'] = output_node

        # Value function estimation stuff
        model['state'] = input_node
        model['target_value'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        model['loss'] = loss = tf.reduce_mean(
            tf.square(tf.squeeze(model['value']) - tf.squeeze(model['target_value'])))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(loss)
        model.add_optimizer(optimizer, loss)
        model['learning_rate'] = lr

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, shape=input_shape)

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph,
                 build_vf_update_feed_dict, name=name)


def cem_model_factory(
        env, network=mlp, network_params={},
        input_shape=None,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        model_file_path=None, name='cem'):
    """
    Model for gradient method
    """

    def build_graph(model, network=network,
                    input_shape=input_shape,
                    network_params=network_params,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', model, network_params=network_params,
            input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model['policy'] = policy
        model.add_output_node(policy.distribution.output_node)

        var_list = policy.get_trainable_variables()
        shapes = map(tf_utils.var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        model['theta'] = tf.placeholder(tf.float32, [total_size])

        var_list = policy.get_trainable_variables()
        model['gf'] = tf_utils.flatten_vars(var_list)
        model['sff'] = tf_utils.setfromflat(var_list, model['theta'])

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph,
                 empty_feed_dict, name=name)


def pg_model_factory(
        env, network=mlp, network_params={}, learning_rate=0.01,
        entropy_weight=0.001, input_shape=None,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        model_file_path=None, name='pg'):
    """
    Model for gradient method
    """

    def build_graph(model, network=network, lr=learning_rate,
                    input_shape=input_shape,
                    network_params=network_params,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', model,
            network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model['policy'] = policy
        model['state'] = model['input:']
        model['Return'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model['output_node'] = policy.distribution.output_node
        model.add_output_node(model['output_node'])

        model['log_pi'] = policy.distribution.log_likelihood(model['action'])
        entropy = tf.reduce_mean(policy.distribution.entropy())

        model['loss'] = -tf.reduce_mean(
            model['log_pi'] * model['Return']) +\
            entropy_weight * entropy

        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.add_loss(model['loss'])
        model.add_optimizer(
            optimizer, model['loss'],
            var_list=policy.get_trainable_variables())

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph, build_pg_update_feed_dict,
                 name=name)


def trpo_model_factory(
        env, network=mlp, network_params={},
        entropy_weight=0,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        input_shape=None, model_file_path=None,
        name='trpo'):
    """
    Policy model for discrete action spaces with policy gradient update
    """
    def build_graph(model, network=network, input_shape=input_shape):

        policy = make_policy(
            env, 'pi', model,
            network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model['policy'] = policy
        old_policy = make_policy(
            env, 'oldpi', model, network_params=network_params,
            input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model['old_policy'] = old_policy

        model['state'] = model['input:']
        model['Return'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model['output_node'] = policy.distribution.output_node
        model.add_output_node(model['output_node'])

        if hasattr(policy.distribution, 'mean'):
            model.add_output_node(
                policy.distribution.mean, name='greedy')

        entropy = tf.reduce_mean(policy.distribution.entropy())
        model['kl'] = tf.reduce_mean(
            old_policy.distribution.kl(policy._distribution))
        entbonus = entropy_weight * entropy

        ratio = policy.distribution.likelihood_ratio(
            model['action'], old_policy.distribution)

        model['surrgain'] = tf.reduce_mean(
            ratio * model['Return'])

        model['optimgain'] = model['surrgain'] + entbonus
        model.G['losses'] = [model['optimgain'], model['kl'],
                             entbonus, model['surrgain'], entropy]

        var_list = policy.get_trainable_variables()
        model.G['klgrads'] = tf.gradients(model['kl'], var_list)

        model['logstd'] = policy.get_trainable_variables()[-1]

        model['pg'] = tf_utils.flatgrad(model['optimgain'], var_list)

        shapes = map(tf_utils.var_shape, var_list)
        start = 0
        tangents = []
        model['flat_tangent'] = tf.placeholder(
            dtype=tf.float32, shape=[None], name='flat_tangent')
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(
                model['flat_tangent'][start:(start + size)], shape)
            tangents.append(param)
            start += size

        model['theta'] = tf.placeholder(tf.float32, [start])

        model['gvp'] = tf.add_n(
            [tf.reduce_sum(g * t)
             for (g, t) in zip(model['klgrads'], tangents)])
        model['fvp'] = tf_utils.flatgrad(model['gvp'], var_list)
        model['gf'] = tf_utils.flatten_vars(var_list)
        model['sff'] = tf_utils.setfromflat(var_list, model['theta'])
        model.G['set_old_pi_eq_new_pi'] = [
            tf.assign(old, new) for (old, new) in
            zip(old_policy.get_variables(), policy.get_variables())]

    if model_file_path is not None:
        return Model.load(model_file_path, name=name)
    return Model(env, build_graph, build_pg_update_feed_dict,
                 name=name)
