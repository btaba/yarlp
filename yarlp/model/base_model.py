# -*- coding: utf-8 -*-
"""
    Base model class
"""

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})


class Model(ABC):
    """
    Abstract class for a function that models
    a policy, value function, or other
    """
    def __init__(self, env):
        self._env = env
        self.create_model()

    @property
    def env(self):
        return self._env

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    @abstractmethod
    def create_model(self):
        self._model = None

    @abstractmethod
    def compile_model(self):
        pass
