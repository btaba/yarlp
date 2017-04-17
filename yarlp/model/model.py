"""Tensorflow model that helps to creates a Graph
"""
import tensorflow as tf
import numpy as np

from yarlp.model.graph import Graph


class Model:
    """
    A Tensorflow model
    """

    def __init__(self, env, build_graph, build_update_feed_dict):
        """
        """
        self._env = env
        self.G = Graph()
        self._loss = None
        self._optimizer = None
        self._optimizer_op = None

        with self.G:
            build_graph(self)
            self.create_weight_setter_ops()
            self.build_update_feed_dict = build_update_feed_dict

    def update(self, *args):
        # this is how we update the weights
        assert self._optimizer_op is not None

        feed_dict = self.build_update_feed(*args)
        _, loss = self.G([self._optimizer_op, self._loss], feed_dict)

        return loss

    def build_update_feed(self, *args):
        """Create the feed dict for self.update
        """
        return self.build_update_feed_dict(self, *args)

    @property
    def env(self):
        return self._env

    def get_env_action_space_dim(self):
        if hasattr(self._env.action_space, 'n'):
            return self._env.action_space.n
        return self._env.action_space.shape[0]

    @property
    def weights(self):
        return self.G.TRAINABLE_VARIABLES

    @weights.setter
    def weights(self, weights):
        feed_dict = {
            self.G['weight_input_var:' + n]: w
            for n, w in weights.items()
        }
        ops = [self.G['set_weight_op:' + n] for n in weights]
        self.G(ops, feed_dict)

    def get_weight_names(self):
        return [w.name for w in self.G.TRAINABLE_VARIABLES]

    def get_weights(self):
        """Get weight values"""
        return self.G(self.weights)

    def run_op(self, var, feed):
        return self.G(var, feed)

    def set_weights(self, weights):
        """Set weights in model from a list of weight values.
        The weight values must be in the same order returned from get_weights()
        """
        weight_dict = {w.name: val for w, val in zip(self.weights, weights)}
        self.weights = weight_dict

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

        assert self._loss is not None
        self._optimizer_op = self._optimizer.minimize(self._loss)

    def add_input(self, name='', dtype=tf.float32, shape=None):
        if shape is None:
            shape = (None, self.env.observation_space.shape[0])

        self.input_node = tf.placeholder(
            dtype=dtype,
            shape=shape)

        self.G['input:' + name] = self.input_node

        return self.input_node

    def add_output(self, network, num_outputs=None, name='', dtype=tf.float32,
                   input_node=None):
        """ Add output node created from network
        """
        if num_outputs is None:
            num_outputs = self.get_env_action_space_dim()

        if input_node is None:
            input_node = self.input_node

        output_node = network(
            inputs=input_node, num_outputs=num_outputs)

        self.G['output:' + name] = output_node

        return output_node

    def add_output_node(self, node, name=''):
        """ Add output node
        """
        self.G['output:' + name] = node
        return node

    def create_weight_setter_ops(self):
        for w in self.weights:
            w_input = tf.placeholder_with_default(w, w.get_shape())
            self.G['weight_input_var:' + w.name] = w_input
            self.G['set_weight_op:' + w.name] = w.assign(w_input)

    def create_gradient_ops_for_node(self,
                                     node, transform_grad_func=lambda x: x):
        grads_and_vars = self.optimizer.compute_gradients(
            node, self.G.TRAINABLE_VARIABLES)

        self.G['gradients:' + node.name] = [
            (transform_grad_func(g), v)
            for g, v in grads_and_vars]

        self.G['gradients_ops:' + node.name] = self.optimizer.apply_gradients(
            self.G['gradients:' + node.name])

    def get_gradients(self, name, feed_dict):
        return self.G(self.G['gradients:' + name], feed_dict)

    def apply_gradient_ops(self, name, feed_dict):
        return self.G(self.G['gradients_ops:' + name], feed_dict)

    def predict(self, data, output_name='output:', input_name='input:'):
        # get the model output for input placeholders
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
        output = self.G[output_name]
        feed_dict = {self.G[input_name]: data}
        return self.G(output, feed_dict).flatten()
