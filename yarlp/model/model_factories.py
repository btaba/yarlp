import numpy as np
import tensorflow as tf

from yarlp.policy.policies import make_policy
from yarlp.model.model import Model
from functools import partial
from yarlp.model.networks import mlp, cnn
from yarlp.utils import tf_utils


def build_vf_update_feed_dict(model, state, target_value, lr=None):
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)
    feed_dict = {model['state']: state,
                 model['target_value']: target_value}
    if lr is not None:
        feed_dict[model['learning_rate']] = lr
    return feed_dict


def empty_feed_dict(*args):
    pass


def build_pg_update_feed_dict(model, state, return_, action, lr=None):
    if len(action.shape) == 1:
        action = action.reshape(-1, 1)
    feed_dict = {model['state']: state,
                 model['Return']: np.squeeze([return_]),
                 model['action']: action}
    if lr is not None:
        feed_dict[model['learning_rate']] = lr
    return feed_dict


def build_a2c_update_feed_dict(model, state, return_, action,
                               advantage, lr=None):
    if len(action.shape) == 1:
        action = action.reshape(-1, 1)
    feed_dict = {model['state']: state,
                 model['Return']: np.squeeze([return_]),
                 model['action']: action,
                 model['advantage']: advantage}
    if lr is not None:
        feed_dict[model['learning_rate']] = lr
    return feed_dict


def build_ddqn_update_feed_dict(
        model, state, action, reward, state_t1,
        done, importance_weights, learning_rate):
    feed_dict = {
        model['state']: state,
        model['action']: action.astype(np.int32),
        model['reward']: reward, model['next_state']: state_t1,
        model['done']: done,
        model['importance_weights']: importance_weights,
        model['learning_rate']: learning_rate
    }
    return feed_dict


def apply_grad_norm_clipping(model, optimizer, tvars, grad_norm_clipping):
    if grad_norm_clipping is not None:
        grad_clipping_func = partial(
            tf.clip_by_global_norm, clip_norm=grad_norm_clipping)
        model.create_gradient_ops_for_node(
            optimizer, model['loss'],
            transform_grad_func=grad_clipping_func,
            tvars=tvars,
            add_optimizer_op=True)
    else:
        model.add_optimizer(
            optimizer, model['loss'],
            var_list=tvars)


def value_function_model_factory(
        env, network=mlp, network_params={},
        learning_rate=0.01, input_shape=None, model_file_path=None,
        grad_norm_clipping=None,
        has_learning_rate_schedule=False,
        name='value_function'):
    """
    Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr, shape):
        input_node = model.add_input(shape=shape)

        network = partial(network, final_activation_fn=None, **network_params)
        output_node = model.add_output(network, num_outputs=1,
                                       input_node=input_node)
        model['value'] = output_node

        # Value function estimation stuff
        model['state'] = input_node
        model['target_value'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        model['squeeze'] = tf.squeeze(model['value']) - tf.squeeze(model['target_value'])
        model['loss'] = loss = 0.5 * tf.reduce_mean(
            tf.square(tf.squeeze(model['value']) -
                      tf.squeeze(model['target_value'])))

        if has_learning_rate_schedule:
            lr = tf.placeholder(tf.float32, (), name="learning_rate")
            model['learning_rate'] = lr

        # optimizer = tf.train.AdamOptimizer(
        #     learning_rate=lr)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=lr, decay=0.99, epsilon=1e-5)

        apply_grad_norm_clipping(model, optimizer,
                                 model.G.TRAINABLE_VARIABLES,
                                 grad_norm_clipping)

        model.add_loss(loss)
        model.add_optimizer(optimizer, loss)

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, shape=input_shape)

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph,
                 build_vf_update_feed_dict, name=name)


def ddqn_model_factory(
        env, network=cnn,
        network_params={},
        double_q=True,
        model_file_path=None, discount_factor=1,
        grad_norm_clipping=10, name='ddqn'):

    def build_graph(model):

        # q network
        q = make_policy(
            env, 'q', model, network_params=network_params, network=network)
        q_vars = q.get_variables()

        # target q network
        q_target = make_policy(
            env, 'q_target', model, network_params=network_params,
            network=network, input_node_name='next_observations')
        q_target_vars = q_target.get_variables()

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
        model['learning_rate'] = tf.placeholder(
            tf.float32, (), name="learning_rate")

        q_target = model['q_target_output']
        q = model['q_output']

        num_actions = model['q_output'].get_shape().as_list()[-1]

        # q values for actions selected
        q_val = tf.reduce_sum(
            model['q_output'] * tf.one_hot(
                tf.squeeze(model['action']), depth=num_actions),
            axis=1)

        # q values for greedy action
        if double_q:
            # user current network to get next greedy action
            with tf.variable_scope('q', reuse=True):
                q_next_state = network(
                    inputs=model['next_state'],
                    num_outputs=num_actions,
                    **network_params)
            q_for_next_state_max = tf.argmax(q_next_state, axis=1)
            q_target_max = tf.reduce_sum(
                (model['q_target_output'] *
                    tf.one_hot(q_for_next_state_max, depth=num_actions)),
                axis=1)
        else:
            q_target_max = tf.reduce_max(model['q_target_output'], axis=1)

        td_return = model['reward'] + \
            discount_factor * q_target_max * (1 - model['done'])
        td_errors = q_val - tf.stop_gradient(td_return)
        # errors = tf.losses.huber_loss(
        #     tf.stop_gradient(td_return), q_val,
        #     reduction=tf.losses.Reduction.NONE)
        model['td_errors'] = td_errors
        errors = 0.5 * tf.square(td_errors)
        weighted_error = tf.reduce_mean(
            model['importance_weights'] * errors)
        model['loss'] = weighted_error
        model.add_loss(model['loss'])

        optimizer = tf.train.AdamOptimizer(
            learning_rate=model['learning_rate'])

        apply_grad_norm_clipping(
            model, optimizer, q_vars, grad_norm_clipping)

        model['update_target_network'] = tf.group(*[
            qt.assign(q) for (q, qt) in
            zip(sorted(q_vars, key=lambda x: x.name),
                sorted(q_target_vars, key=lambda x: x.name))])

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


def a2c_model_factory(
        env, network=mlp, policy_network_params={},
        value_network_params={}, learning_rate=0.01,
        has_learning_rate_schedule=False,
        entropy_weight=0.01,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        grad_norm_clipping=None,
        model_file_path=None, name='a2c'):

    def build_graph(model, network=network, lr=learning_rate,
                    init_std=init_std, adaptive_std=adaptive_std):

        policy = make_policy(
            env, 'pi', model,
            network_params=policy_network_params,
            init_std=init_std, adaptive_std=adaptive_std, network=network)

        model['policy'] = policy
        model['state'] = model['input:observations']
        model['advantage'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='advantage')
        model['Return'] = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model['output_node'] = policy.distribution.output_node
        model.add_output_node(model['output_node'])

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            vf = network(
                inputs=model['state'],
                num_outputs=1, final_scope='vf',
                **value_network_params)
        model['vf'] = vf
        model['log_pi'] = policy.distribution.log_likelihood(model['action'])
        entropy = tf.reduce_mean(policy.distribution.entropy())
        model['entropy'] = entropy

        model['vf_loss'] = 0.5 * tf.reduce_mean(
            tf.square(tf.squeeze(vf) - model['Return']))
        model['loss'] = -tf.reduce_mean(
            model['log_pi'] * model['advantage']) -\
            entropy_weight * entropy + model['vf_loss']

        model.add_loss(model['loss'])

        if has_learning_rate_schedule:
            lr = tf.placeholder(tf.float32, (), name="learning_rate")
            model['learning_rate'] = lr

        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=lr, epsilon=1e-5, decay=0.99)

        apply_grad_norm_clipping(
            model, optimizer, policy.get_trainable_variables(),
            grad_norm_clipping)

    if model_file_path is not None:
        return Model.load(model_file_path, name)
    return Model(env, build_graph, build_a2c_update_feed_dict,
                 name=name)


def pg_model_factory(
        env, network=mlp, network_params={}, learning_rate=0.01,
        has_learning_rate_schedule=False,
        entropy_weight=0.001,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        grad_norm_clipping=None,
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
        model['entropy'] = entropy

        model['loss'] = -tf.reduce_mean(
            model['log_pi'] * model['Return']) -\
            entropy_weight * entropy
        model.add_loss(model['loss'])
        if has_learning_rate_schedule:
            lr = tf.placeholder(tf.float32, (), name="learning_rate")
            model['learning_rate'] = lr

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        apply_grad_norm_clipping(
            model, optimizer, policy.get_trainable_variables(),
            grad_norm_clipping)

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
