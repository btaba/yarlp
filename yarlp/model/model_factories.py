import numpy as np
import tensorflow as tf

from yarlp.model.tf_model import Model
from functools import partial


def value_function_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01, input_shape=None):
    """Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr, shape):
        input_node = model.add_input(shape=shape)

        vsi = tf.contrib.layers.variance_scaling_initializer()
        network = partial(network, activation_fn=None,
                          weights_initializer=vsi)
        output_node = model.add_output(network, num_outputs=1)

        # Value function estimation stuff
        model.state = input_node
        model.target_value = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        model.loss = tf.squared_difference(output_node, model.target_value)
        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        model.learning_rate = lr

        model.value = output_node
        model.create_gradient_ops_for_node(output_node)

    def build_update_feed_dict(model, state, target_value):
        feed_dict = {model.state: np.expand_dims(state, 0),
                     model.target_value: [target_value]}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, shape=input_shape)

    return Model(env, build_graph, build_update_feed_dict)


def discrete_pg_model_factory(
        env, network, learning_rate=0.01, input_shape=None):
    """Policy model for discrete action spaces with policy gradient update
    """
    def build_graph(model, network, lr, input_shape):
        input_node = model.add_input(shape=input_shape)

        model.state = input_node
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.learning_rate = lr

        # Softmax policy for discrete action spaces
        network = partial(network, activation_fn=tf.nn.softmax)
        output_node = model.add_output(network)
        model.action = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='action')
        action_probability = tf.gather(
            tf.squeeze(output_node), model.action)

        model.loss = -tf.log(action_probability) * model.Return
        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)

        model.log_pi = tf.log(action_probability)
        model.create_gradient_ops_for_node(model.log_pi)

    def build_update_feed_dict(model, state, return_, action):
        feed_dict = {model.state: np.expand_dims(np.array(state), 0),
                     model.Return: [return_], model.action: [action]}
        return feed_dict

    build_graph = partial(build_graph, network=network,
                          lr=learning_rate, input_shape=input_shape)

    return Model(env, build_graph, build_update_feed_dict)


def continuous_gaussian_pg_model_factory(
        env, learning_rate=0.01, input_shape=None):
    """Policy model for continuous action spaces with scaled policy gradient update

    [1] Degris, T., Pilarski, P., & Sutton, R. (2012). Model-free reinforcement
    learning with continuous action in practice. … Control Conference (ACC),
    2177–2182. doi:10.1109/ACC.2012.6315022

    Gradients are scaled with sigma**2
    due to numerical instability mentioned in [1]
    """
    def build_graph(model, learning_rate, input_shape):
        network = tf.contrib.layers.fully_connected
        input_node = model.add_input(shape=input_shape)

        model.state = input_node
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.learning_rate = learning_rate

        network = partial(network, activation_fn=None)
        model.mu = model.add_output(network, name='mean')

        model.sigma = model.add_output(network, name='std_dev')
        model.sigma = tf.exp(model.sigma) + 1e-5

        model.normal_dist = tf.contrib.distributions.Normal(
            model.mu, model.sigma)
        model.action = tf.squeeze(model.normal_dist.sample([1]))
        model.action = tf.clip_by_value(
            model.action, model._env.action_space.low[0],
            model._env.action_space.high[0])
        model.add_output_node(model.action)

        model.log_pi = model.normal_dist.log_prob(model.action)
        model.loss = -model.log_pi * model.Return

        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

    def build_update_feed_dict(model, state, return_, action):
        feed_dict = {model.state: np.expand_dims(np.array(state), 0),
                     model.Return: [return_], model.action: [action]}
        return feed_dict

    class ScaledNormal(Model):
        def get_gradients(self, node_name, feed):
            weights = self.get_weights()
            mu = self.run_op(self.mu, feed)
            sigma = self.run_op(self.sigma, feed)
            state = feed[self.state]
            action = feed[self.action.name]

            grads = [
                (action - mu) * state.T,
                weights[1] * 0 + (action - mu),
                ((action - mu)**2 - sigma**2) * state.T,
                weights[3] * 0 + ((action - mu)**2 - sigma**2)
            ]
            return grads

    build_graph = partial(build_graph,
                          learning_rate=learning_rate,
                          input_shape=input_shape)
    # m = Model(env, build_graph, build_update_feed_dict)
    m = ScaledNormal(env, build_graph, build_update_feed_dict)

    return m
