"""
Tensorflow framework for models
"""

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})


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
        self._sess.run(
            tf.variables_initializer(self.global_variables)
        )
        self._graph.finalize()
        self._context.__exit__(*args)

    def __setitem__(self, var_name, tf_node):
        # Collections are not sets, so it's possible to add several times
        if var_name in self:
            raise KeyError('%s is already in the graph' % var_name)
        self._graph.add_to_collection(var_name, tf_node)

    def __getitem__(self, var_names):
        if isinstance(var_names, list):
            return self._graph.get_collection(var_names)
        return self._graph.get_collection(var_names)[0]

    def __call__(self, operations, feed_dict):
        self._session.run(operations, feed_dict)

    @property
    def global_variables(self):
        return self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    @property
    def trainable_variables(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


class Model:
    """
    A Tensorflow model
    """

    def __init__(self, network):
        self._network = network
        self.G = Graph()

        with self.G as g:
            self.model_builder()
            self.create_weight_setter_operations()

    @abstractmethod
    def model_builder(self):
        # this is where the specs of the model live
        pass

    @abstractmethod
    def update(self, feed_dict):
        # this is where we update the weights
        pass

        @abstractmethod
    def get_action(self):
        # this is where we predict the next action to take
        pass

    @property
    def weights(self):
        return self.G[self.G.trainable_variables()]

    @weights.setter
    def weights(self, weights):
        feed_dict = {
            self.G['weight_input_var:' + n]: w
            for n, w in weights.items()
        }
        operations = [self.G['set_weight:' + n] for n in weights]
        self.G(operations, feed_dict)

    def create_weight_setter_operations(self):
        for w in self.weights:
            w_input = tf.placeholder_with_default(w, w.get_shape())
            self.G['weight_input_var:' + w.name] = w_input
            self.G['set_weight:' + w.name] = w.assign(var_placeholder)
