"""
    Regression tests for the Graph
"""

import os
import unittest
import shutil
import tensorflow as tf
from yarlp.model.graph import Graph


class TestGraph(unittest.TestCase):

    def test_setitem(self):
        tf.reset_default_graph()
        G = Graph()
        var = tf.placeholder(dtype=tf.float32, shape=(None,))
        G['var'] = var

        with self.assertRaises(KeyError):
            G['var'] = var

    def test_getitem(self):
        tf.reset_default_graph()
        G = Graph()
        var = tf.placeholder(dtype=tf.float32, shape=(None,))
        G['var'] = var

        self.assertEqual(G['var'], var)
        self.assertEqual(G[['var', 'var']], [var, var])

    def test_contains(self):
        tf.reset_default_graph()
        G = Graph()
        var = tf.placeholder(dtype=tf.float32, shape=(None,))
        G['var'] = var
        self.assertTrue('var' in G)

    def test_global_vars(self):
        tf.reset_default_graph()
        G = Graph()
        with G as g:
            var = tf.Variable(tf.random_normal([10, 10]), trainable=False)
            g['var'] = var

        self.assertEqual(g.GLOBAL_VARIABLES, [var])

    def test_trainable_vars(self):
        tf.reset_default_graph()
        G = Graph()
        with G as g:
            var = tf.Variable(tf.random_normal([10, 10]), trainable=True)
            g['var'] = var

        self.assertEqual(g.TRAINABLE_VARIABLES, [var])


class TestGraphSaveLoad(unittest.TestCase):

    def test_load_and_save(self):
        tf.reset_default_graph()
        G = Graph()
        with G as g:
            var = tf.Variable(tf.random_normal([10, 10]), trainable=False)
            g['var'] = var

        G.save('test_load_and_save')
        tf.reset_default_graph()
        G = Graph()
        G.load('test_load_and_save')
        assert 'var' in G

    def tearDown(self):
        if os.path.exists('test_load_and_save'):
            shutil.rmtree('test_load_and_save')
