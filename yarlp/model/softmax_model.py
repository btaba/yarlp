# -*- coding: utf-8 -*-
"""
    Single layer softmax regression
"""

from yarlp.model.base_model import Model
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import numpy as np


class KerasSoftmaxModel(Model):
    """
    A Model class implementing a 1-layer softmax function
    """
    def __init__(self, env):
        super().__init__(env)
        self.compiled = False

    def create_model(self):
        self._model = Sequential()
        self._model.add(
            Dense(
                input_dim=self._env.observation_space.shape[0],
                output_dim=self._env.action_space.n,
                activation="softmax"
            )
        )

    def compile_model(self):
        self.model.compile(optimizer='sgd', loss='mse')
        self.compiled = True

    def get_weights(self):
        return self._model.get_weights()

    def predict_on_batch(self, data):
        return self._model.predict_on_batch(data)

    def set_weights(self, weights):
        return self._model.set_weights(weights)


class TFSoftmaxModel:
    """
    TF Softmax Model - demo
    Inspired by https://github.com/danijar/mindpark
    """

    def __init__(self, env):
        # super().__init__(env)
        self._env = env
        self._graph = tf.Graph()
        self._sess = tf.Session('', graph=self._graph)

        # build the graph
        with self._graph.as_default():
            # state
            input_node = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self._env.observation_space.shape[0]))
            self._graph.add_to_collection('input/', input_node)

            output_node = tf.contrib.layers.fully_connected(
                inputs=input_node, num_outputs=self._env.action_space.n,
                activation_fn=tf.nn.softmax)
            self._graph.add_to_collection('output/', output_node)

            # Policy gradient Stuff
            self.state = input_node
            self.action = tf.placeholder(dtype=tf.int32, shape=(None,), name='action')
            action_probability = tf.gather(tf.squeeze(output_node), self.action)
            self.return_ = tf.placeholder(dtype=tf.float32, shape=(None,), name='return')
            self.loss = -tf.log(action_probability) * self.return_
            self.optimizer = tf.train.AdamOptimizer(learning_rate=.01)
            self.train_operation = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            # # Make TF operations for setting the weights in the graph
            for var in self.weights:
                var_placeholder = tf.placeholder_with_default(var, var.get_shape())
                self._graph.add_to_collection('set_weight_input/' + var.name, var_placeholder)
                self._graph.add_to_collection('set_weight/' + var.name, var.assign(var_placeholder))

            init_op = tf.variables_initializer(self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)
            self._graph.finalize()

    def update(self, state, return_, action):
        # print(np.expand_dims(state, 0), return_, action)
        feed_dict = {self.state: np.array(np.expand_dims(state, 0)), self.return_: [return_], self.action: [action]}
        _, loss = self._sess.run([self.train_operation, self.loss], feed_dict)
        # print(loss)
        return loss

    @property
    def weights(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @weights.setter
    def weights(self, weights):
        self.validate_weights(weights)
        feed_dict = {'set_weight_input/' + name: value
                     for name, value in weights.items()}
        ops = ['set_weight/' + x for x in weights]
        feed_dict = {self.get_collection(k): v for k, v in feed_dict.items()}
        ops = [self.get_collection(o) for o in ops]
        return self._sess.run(ops, feed_dict)

    def get_collection(self, name):
        # This can be a __getitem__ if I were to make Graph object
        return self._graph.get_collection(name)[0]

    def set_weights(self, weights):
        """Set weights in model from a list"""
        weight_dict = {w.name: val for w, val in zip(self.weights, weights)}
        self.weights = weight_dict

    def get_weights(self):
        return self._sess.run(self.weights)

    def get_weights_dict(self):
        """Get weights in a dict with the weight names as keys and arrays as values
        """
        return {w.name: val for w, val in
                zip(self.weights, self.get_weights())}

    def validate_weights(self, weights):
        for name in weights:
            if name not in self.weight_names:
                raise KeyError('unrecognized weight name ' + name)

    @property
    def weight_names(self):
        return {x.name for x in self.weights}

    def predict_on_batch(self, data):
        # data should be a numpy array
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
        output = self.get_collection('output/')
        feed_dict = {self.get_collection('input/'): data}
        return self._sess.run(output, feed_dict)
