"""
    Regression tests for the Model
"""

import gym
import shutil
from yarlp.model.model_factories import value_function_model_factory
from yarlp.model.model_factories import cem_model_factory
from yarlp.model.model_factories import pg_model_factory
from yarlp.model.model_factories import trpo_model_factory


def test_value_function():
    env = gym.make('CartPole-v0')
    M = value_function_model_factory(env)
    M.save('test_load_and_save_value_func')
    del M
    value_function_model_factory(
        env, model_file_path='test_load_and_save_value_func')
    shutil.rmtree('test_load_and_save_value_func')


def test_cem_function():
    env = gym.make('CartPole-v0')
    M = cem_model_factory(env)
    M.save('test_load_and_save_cem_func')
    del M
    cem_model_factory(
        env, model_file_path='test_load_and_save_cem_func')
    shutil.rmtree('test_load_and_save_cem_func')


def test_cem_function_continuous():
    env = gym.make('MountainCarContinuous-v0')
    M = cem_model_factory(env)
    M.save('test_load_and_save_cem_func1')
    del M
    cem_model_factory(
        env, model_file_path='test_load_and_save_cem_func1')
    shutil.rmtree('test_load_and_save_cem_func1')


def test_pg_function():
    env = gym.make('CartPole-v0')
    M = pg_model_factory(env)
    M.save('test_load_and_save_pg')
    del M
    pg_model_factory(
        env, model_file_path='test_load_and_save_pg')
    shutil.rmtree('test_load_and_save_pg')


def test_pg_function_continuous():
    env = gym.make('MountainCarContinuous-v0')
    M = pg_model_factory(env)
    M.save('test_load_and_save_pg1')
    del M
    pg_model_factory(
        env, model_file_path='test_load_and_save_pg1')
    shutil.rmtree('test_load_and_save_pg1')


def test_trpo_function():
    env = gym.make('CartPole-v0')
    M = trpo_model_factory(env)
    M.save('test_load_and_save_trpo')
    del M
    trpo_model_factory(
        env, model_file_path='test_load_and_save_trpo')
    shutil.rmtree('test_load_and_save_trpo')


def test_trpo_function_continuous():
    env = gym.make('MountainCarContinuous-v0')
    M = trpo_model_factory(env)
    M.save('test_load_and_save_trpo1')
    del M
    trpo_model_factory(
        env, model_file_path='test_load_and_save_trpo1')
    shutil.rmtree('test_load_and_save_trpo1')
