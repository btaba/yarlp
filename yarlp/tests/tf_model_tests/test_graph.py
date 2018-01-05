"""
    Regression tests for the Graph
"""

import os
import pytest
import shutil
import numpy as np
import tensorflow as tf
from yarlp.model.graph import Graph


def test_setitem():
    tf.reset_default_graph()
    G = Graph()
    var = tf.placeholder(dtype=tf.float32, shape=(None,))
    G['var'] = var

    with pytest.raises(KeyError):
        G['var'] = var


def test_getitem():
    tf.reset_default_graph()
    G = Graph()
    var = tf.placeholder(dtype=tf.float32, shape=(None,))
    G['var'] = var

    assert G['var'] == var
    assert G[['var', 'var']] == [var, var]


def test_contains():
    tf.reset_default_graph()
    G = Graph()
    var = tf.placeholder(dtype=tf.float32, shape=(None,))
    G['var'] = var
    assert 'var' in G


def test_global_vars():
    tf.reset_default_graph()
    G = Graph()
    with G as g:
        var = tf.Variable(tf.random_normal([10, 10]), trainable=False)
        g['var'] = var

    assert g.GLOBAL_VARIABLES == [var]


def test_trainable_vars():
    tf.reset_default_graph()
    G = Graph()
    with G as g:
        var = tf.Variable(tf.random_normal([10, 10]), trainable=True)
        g['var'] = var

    assert g.TRAINABLE_VARIABLES == [var]


def test_load_and_save():
    try:
        rand_var = np.random.random([10, 10])
        tf.reset_default_graph()
        G = Graph()
        with G as g:
            var = tf.Variable(rand_var, trainable=False)
            g['var'] = var

        G.save('test_load_and_save')
        tf.reset_default_graph()
        G = Graph()
        G.load('test_load_and_save')
        assert 'var' in G

        with G._session.as_default() as sess:
            assert np.allclose(rand_var, sess.run(G['var']))
    except Exception as e:
        raise e
    else:
        if os.path.exists('test_load_and_save'):
            shutil.rmtree('test_load_and_save')
