import numpy as np
import tensorflow as tf

from yarlp.model.tf_model import Model
from functools import partial


def cem_model_factory(env, network=tf.contrib.layers.fully_connected):

    def build_network(model, network):
        model.add_input()
        from functools import partial
        network = partial(network, activation_fn=tf.nn.softmax)
        model.add_output(network)

    def update(model):
        pass

    build_network = partial(build_network, network=network)

    return Model(env, build_network, update)


def value_function_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01):

    def build_network(model, network):
        input_node = model.add_input()
        from functools import partial
        network = partial(network, activation_fn=tf.nn.softmax)
        output_node = model.add_output(network, num_outputs=1)

        # Value function estimation stuff
        model.state = input_node
        model.target_value = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='target_value')
        model.loss = tf.squared_difference(output_node, model.target_value)
        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        model.optimize_operation = model.optimizer.minimize(model.loss)

    def update(model, state, target_value):
        feed_dict = {model.state: np.expand_dims(state, 0),
                     model.target_value: [target_value]}
        _, loss = model.G([model.optimize_operation, model.loss], feed_dict)
        return loss

    build_network = partial(build_network, network=network)

    return Model(env, build_network, update)


def policy_gradient_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01):

    def build_network(model, network):
        input_node = model.add_input()
        from functools import partial
        network = partial(network, activation_fn=tf.nn.softmax)
        output_node = model.add_output(network)

        # Policy gradient stuff
        model.state = input_node
        model.action = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='action')
        action_probability = tf.gather(tf.squeeze(output_node), model.action)
        model.Return = tf.placeholder(
            dtype=tf.float32, shape=(None,), name='return')
        model.loss = -tf.log(action_probability) * model.Return
        model.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        model.optimize_operation = model.optimizer.minimize(model.loss)

    def update(model, state, return_, action):
        # this is where we update the weights
        feed_dict = {model.state: np.array(np.expand_dims(state, 0)),
                     model.Return: [return_], model.action: [action]}
        _, loss = model.G([model.optimize_operation, model.loss], feed_dict)
        return loss

    build_network = partial(build_network, network=network)

    return Model(env, build_network, update)
