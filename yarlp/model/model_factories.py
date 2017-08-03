import numpy as np
import tensorflow as tf

from yarlp.policies.policies import make_policy
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
            env, network_params=network_params, input_shape=input_shape,
            init_std=init_std, adaptive_std=adaptive_std, network=network)

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
        model.add_optimizer(model.optimizer, model.loss)

    def build_update_feed_dict(model, state, return_, action):
        feed_dict = {model.state: state,
                     model.Return: np.squeeze([return_]), model.action: action}
        return feed_dict

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)


def discrete_trpo_model_factory(
        env, network, learning_rate=0.01, entropy_weight=0.001,
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
        model.probs = model.output_node
        model.action = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='action')
        action_one_hot = tf.one_hot(model.action, model.output_node.shape[1])
        model.action_one_hot = action_one_hot

        model.pi = tf.squeeze(tf.reduce_sum(
            action_one_hot * model.output_node, 1))
        model.log_pi = tf.squeeze(tf.log(model.pi + tf_utils.EPSILON))

        shape = model.output_node.get_shape().as_list()
        model.old_pi_placeholder = tf.placeholder(
            dtype=tf.float32, shape=shape, name='old_pi')

        model.old_pi = tf.squeeze(tf.reduce_sum(
            action_one_hot * model.old_pi_placeholder, 1))
        model.logli_old = tf.squeeze(tf.log(model.old_pi + tf_utils.EPSILON))

        model.lr = tf.exp(model.log_pi - model.logli_old)

        # get surrogate loss function
        var_list = list(tf.trainable_variables())
        entropy = -tf.reduce_mean(
            tf.reduce_sum(
                model.probs * tf.log(model.probs + tf_utils.EPSILON), axis=-1)
        )

        # TODO: add entropy to loss function
        model.surr_loss = -tf.reduce_mean(
            model.lr * model.Return)

        model.kl = tf.reduce_mean(
            tf.reduce_sum(model.old_pi_placeholder * (
                tf.log(model.old_pi_placeholder + tf_utils.EPSILON) -
                tf.log(model.probs + tf_utils.EPSILON)), axis=-1)
        )

        # KL divergence where first arg is fixed
        # model.probs_fixed = tf.stop_gradient(model.probs)
        # model.kl_firstfixed = tf.reduce_mean(
        #     tf.reduce_sum(model.probs_fixed * (
        #         tf.log(model.probs_fixed + tf_utils.EPSILON) -
        #         tf.log(model.probs + tf_utils.EPSILON)), axis=-1)
        # )

        model.grads = tf.gradients(model.kl, var_list)
        model.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None])
        model.pg = tf_utils.flatgrad(model.surr_loss, var_list)

        model.losses = [model.surr_loss, model.kl, entropy]

        shapes = map(tf_utils.var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(model.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size

        shapes = map(tf_utils.var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        model.theta = tf.placeholder(tf.float32, [total_size])

        # gradient vector product
        model.gvp = tf.add_n(
            [tf.reduce_sum(g * t) for (g, t) in zip(model.grads, tangents)])
        model.fvp = tf_utils.flatgrad(model.gvp, var_list)
        model.gf = tf_utils.flatten_vars(var_list)
        model.sff = tf_utils.setfromflat(var_list, model.theta)

    def build_update_feed_dict(model, state, return_, action, old_pi):
        feed_dict = {model.state: state,
                     model.Return: np.squeeze([return_]), model.action: action,
                     model.old_pi_placeholder: old_pi}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, input_shape=input_shape)

    if model_file_path is not None:
        return Model(env, None, build_update_feed_dict, model_file_path)
    return Model(env, build_graph, build_update_feed_dict)
