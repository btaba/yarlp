# -*- coding: utf-8 -*-
"""
    Base Agent class, which takes in a PolicyModel object
"""

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """
    def __init__(self, env):
        self._env = env
        # self._policy = policy_model
        # self._value = value_function
        # self._env = self._policy._env

    @property
    def env(self):
        return self._env

    # @property
    # def policy(self):
    #     return self._policy

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_action(self):
        pass
