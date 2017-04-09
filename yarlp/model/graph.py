"""
Tensorflow Graph helper class
"""

import tensorflow as tf


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

    def __call__(self, ops, feed_dict={}):
        return self._session.run(ops, feed_dict)

    def trainable_variables_for_scope(self, scope):
        return self._graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    @property
    def GLOBAL_VARIABLES(self):
        return self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    @property
    def TRAINABLE_VARIABLES(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
