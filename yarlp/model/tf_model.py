"""
Tensorflow framework for models
"""

import tensorflow as tf
import numpy as np


class Graph:
    """
    Tensorflow Graph interface
    """

    def __init__(self):
        self._graph = tf.Graph()
        self._session = tf.Session('', graph=self._graph)

    def __enter__(self):
        self._context = self._graph.as_default()
        self._context.__enter__()
        return self

    def __exit__(self, *args):
        self._session.run(
            tf.variables_initializer(self.GLOBAL_VARIABLES)
        )
        self._graph.finalize()
        self._context.__exit__(*args)

    def __contains__(self, var_name):
        return var_name in self._graph.get_all_collection_keys()

    def __setitem__(self, var_name, tf_node):
        # Collections are not sets, so it's possible to add several times
        if var_name in self:
            raise KeyError('"%s" is already in the graph.' % var_name)
        self._graph.add_to_collection(var_name, tf_node)

    def __getitem__(self, var_names):
        if isinstance(var_names, list):
            return [self[v] for v in var_names]

        if var_names not in self:
            raise KeyError('"%s" does not exist in the graph.' % var_names)
        return self._graph.get_collection(var_names)[0]

    def __call__(self, operations, feed_dict={}):
        return self._session.run(operations, feed_dict)

    @property
    def GLOBAL_VARIABLES(self):
        return self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    @property
    def TRAINABLE_VARIABLES(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


class Model:
    """
    A Tensorflow model
    """

    def __init__(self, env, build_network, update_func):
        self._env = env
        self.G = Graph()

        with self.G:
            build_network(self)
            self.create_weight_setter_operations()
            self.update_func = update_func

    def update(self, *args):
        # this is where we update the weights
        return self.update_func(*args)

    @property
    def env(self):
        return self._env

    @property
    def weights(self):
        return self.G.TRAINABLE_VARIABLES

    @weights.setter
    def weights(self, weights):
        feed_dict = {
            self.G['weight_input_var:' + n]: w
            for n, w in weights.items()
        }
        operations = [self.G['set_weight_op:' + n] for n in weights]
        self.G(operations, feed_dict)

    def get_weight_names(self):
        return [w.name for w in self.G.TRAINABLE_VARIABLES]

    def get_weights(self):
        """ get weight values """
        return self.G(self.weights)

    def set_weights(self, weights):
        """Set weights in model from a list of weight values"""
        weight_dict = {w.name: val for w, val in zip(self.weights, weights)}
        self.weights = weight_dict

    def add_input(self, name='', dtype=tf.float32, shape=None):
        if shape is None:
            shape = (None, self.env.observation_space.shape[0])

        self.input_node = tf.placeholder(
            dtype=dtype,
            shape=shape)

        self.G['input:' + name] = self.input_node

        return self.input_node

    def add_output(self, network, name='', dtype=tf.float32, num_outputs=None):
        if num_outputs is None:
            num_outputs = self._env.action_space.n

        self.output_node = network(
            inputs=self.input_node, num_outputs=num_outputs)

        self.G['output:' + name] = self.output_node

        return self.output_node

    def create_weight_setter_operations(self):
        for w in self.weights:
            w_input = tf.placeholder_with_default(w, w.get_shape())
            self.G['weight_input_var:' + w.name] = w_input
            self.G['set_weight_op:' + w.name] = w.assign(w_input)

    def get_action(self, data, output_name='output:', input_name='input:'):
        # this is where we predict the next action to take
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
        output = self.G[output_name]
        feed_dict = {self.G[input_name]: data}
        return self.G(output, feed_dict)


def policy_gradient_model_factory(env, learning_rate=0.01):

    def build_network(model):
        input_node = model.add_input()
        from functools import partial
        network = partial(
            tf.contrib.layers.fully_connected, activation_fn=tf.nn.softmax)
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

    return Model(env, build_network, update)
