"""
Tensorflow Graph helper class
"""
import os
import tensorflow as tf


class Graph:
    """
    Tensorflow Graph interface
    """

    def __init__(self):
        self._graph = tf.Graph()
        self._graph.seed = tf.get_default_graph().seed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._session = tf.Session('', graph=self._graph, config=config)
        self._saver = None

    def __enter__(self):
        self._context = self._graph.as_default()
        self._context.__enter__()
        return self

    def __exit__(self, *args):
        self._session.run(
            tf.variables_initializer(self.GLOBAL_VARIABLES)
        )
        self._saver = tf.train.Saver()
        # if self._finalize:
        #     self._graph.finalize()
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

    def save(self, path):
        path = self._get_clean_path(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self._session, path)

    def load(self, path):
        path = self._get_clean_path(path)
        assert os.path.isdir(path)
        # saver = tf.train.Saver()
        with self._graph.as_default():
            self._saver = tf.train.import_meta_graph(path + '.meta')
            self._saver.restore(self._session, path)
        # self._graph.finalize()

    def _get_clean_path(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        path = os.path.join(path, 'graph.ckpt')
        return path
