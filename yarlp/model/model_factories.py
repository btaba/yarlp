import numpy as np
import tensorflow as tf

from yarlp.model.tf_model import Model
from functools import partial


def value_function_model_factory(
        env, network=tf.contrib.layers.fully_connected,
        learning_rate=0.01):
    """ Minimizes squared error of state-value function
    """

    def build_graph(model, network, lr):
        input_node = model.add_input()

        network = partial(network, activation_fn=None)
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
                          lr=learning_rate)

    return Model(env, build_graph, build_update_feed_dict)
