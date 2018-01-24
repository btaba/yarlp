import numpy as np
import tensorflow as tf

from yarlp.policy.policies import make_policy
from yarlp.model.model import Model
from functools import partial
from yarlp.model.networks import mlp, cnn
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


def build_ddqn_update_feed_dict(
        model, state, action, reward, state_t1,
        done, importance_weights):
    feed_dict = {
        model['state']: state, model['action']: action.astype(np.int32),
        model['reward']: reward, model['next_state']: state_t1,
        model['done']: done,
        model['importance_weights']: importance_weights
    }
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


def ddqn_model_factory(
        env, network=cnn,
        network_params={},
        double_q=True, learning_rate=5e-4,
        model_file_path=None, discount_factor=1,
        grad_norm_clipping=10, name='ddqn'):

    def build_graph(model):

        # q network
        q = make_policy(
            env, 'q', model, network_params=network_params, network=network)
        q_vars = q.get_trainable_variables()

        # target q network
        q_target = make_policy(
            env, 'q_target', model, network_params=network_params,
            network=network, input_node_name='next_observations')
        q_target_vars = q_target.get_trainable_variables()

        model['q'] = q
        model['q_target'] = q_target
        model['q_output'] = model['q:logits']
        model['q_target_output'] = model['q_target:logits']
        model['state'] = model['input:observations']
        model['next_state'] = model['input:next_observations']
        model['reward'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='reward')
        model['done'] = tf.placeholder(tf.float32, (None,), name='done')
        model['importance_weights'] = tf.placeholder(
            tf.float32, (None,), name='imp_weights')

        num_actions = tf.shape(model['q_output'])[-1]
        # q values for actions selected
        q_val = tf.reduce_sum(
            model['q_output'] * tf.one_hot(
                tf.squeeze(model['action']),
                depth=num_actions),
            axis=1)

        # q values for greedy action
        if double_q:
            # user current network to get next greedy action
            with tf.variable_scope('q', reuse=True):
                q_next_state = network(
                    inputs=model['next_state'],
                    num_outputs=model['q_output'].get_shape().as_list()[-1],
                    **network_params)
            q_for_next_state_max = tf.argmax(q_next_state, axis=1)
            q_target_max = tf.reduce_sum(
                (model['q_target_output'] *
                    tf.one_hot(q_for_next_state_max, depth=num_actions)),
                axis=1
            )
        else:
            q_target_max = tf.reduce_max(model['q_target_output'], 1)

        td_return = model['reward'] + \
            discount_factor * q_target_max * (1 - model['done'])
        td_errors = q_val - tf.stop_gradient(td_return)
        model['td_errors'] = td_errors
        errors = tf_utils.huber_loss(td_errors)
        weighted_error = tf.reduce_mean(model['importance_weights'] * errors)
        model['loss'] = weighted_error
        model.add_loss(model['loss'])

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.99, epsilon=0.01)
        if grad_norm_clipping is not None:
            grad_clipping_func = partial(
                tf.clip_by_norm, clip_norm=grad_norm_clipping)
            model.create_gradient_ops_for_node(
                optimizer, model['loss'],
                transform_grad_func=grad_clipping_func,
                tvars=q_vars, add_optimizer_op=True)
        else:
            model.add_optimizer(
                optimizer, model['loss'],
                var_list=q_vars)

        model['update_target_network'] = tf.group(*[
            qt.assign(q) for (q, qt) in
            zip(q_vars, q_target_vars)])

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph,
                 build_ddqn_update_feed_dict, name=name)


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
                    network_params=network_params):

        policy = make_policy(
            env, 'pi', model, network_params=network_params,
            input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std,
            min_std=min_std, network=network)
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
        entropy_weight=0.001,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        model_file_path=None, name='pg'):
    """
    Model for gradient method
    """

    def build_graph(model, network=network, lr=learning_rate,
                    network_params=network_params,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', model,
            network_params=network_params,
            init_std=init_std, adaptive_std=adaptive_std, network=network)
        model['policy'] = policy
        model['state'] = model['input:observations']
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

        model['state'] = model['input:observations']
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
        model['losses'] = tf.stack([model['optimgain'], model['kl'],
                                    entbonus, model['surrgain'], entropy])

        var_list = policy.get_trainable_variables()
        klgrads = tf.gradients(model['kl'], var_list)

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
             for (g, t) in zip(klgrads, tangents)])
        model['fvp'] = tf_utils.flatgrad(model['gvp'], var_list)
        model['gf'] = tf_utils.flatten_vars(var_list)
        model['sff'] = tf_utils.setfromflat(var_list, model['theta'])
        model['set_old_pi_eq_new_pi'] = tf.group(*[
            tf.assign(old, new) for (old, new) in
            zip(old_policy.get_variables(), policy.get_variables())])

    if model_file_path is not None:
        return Model.load(model_file_path, name=name)
    return Model(env, build_graph, build_pg_update_feed_dict,
                 name=name)
