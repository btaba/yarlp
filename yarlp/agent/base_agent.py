"""
    Base Agent class, which takes in a PolicyModel object
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from yarlp.utils.env_utils import env_action_space_is_discrete
from yarlp.utils.env_utils import get_env_action_space_dim
from yarlp.utils.metric_logger import MetricLogger
from yarlp.utils.replay_buffer import Rollout


ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """

    def __init__(self, env, discount_factor=1,
                 logger=None,
                 state_featurizer=lambda x: x):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        self._env = env

        # Discount factor
        assert discount_factor >= 0 and discount_factor <= 1
        self._discount = discount_factor

        if logger is None:
            self.logger = MetricLogger()
        else:
            self.logger = logger

        self._state_featurizer = state_featurizer

    @abstractmethod
    def train(self):
        pass

    @property
    def env(self):
        return self._env

    @property
    def num_actions(self):
        return get_env_action_space_dim(self._env)

    def rollout(self, render=False, render_freq=5):
        """
        Performs actions on the environment
        based on the agent's current weights

        Returns
        ----------
        Rollout : named tuple
        """
        r = Rollout([], [], [])

        observation = self._env.reset()
        observation = self.get_state(observation)
        for t in range(self._env.spec.timestep_limit):
            r.states.append(observation)
            action = self.get_action(observation)
            (observation, reward, done, _) = self._env.step(action)

            if render and t and t % render_freq == 0:
                self._env.render()

            observation = self.get_state(observation)
            r.rewards.append(reward)
            r.actions.append(action)
            if done:
                break

        return r

    def do_greedy_episode(self):
        r = Rollout([], [], [])

        t = 0
        done = False
        observation = self._env.reset()
        observation = self.get_state(observation)
        while not done:
            r.states.append(observation)
            action = self.get_action(observation, greedy=True)
            (observation, reward, done, _) = self._env.step(action)
            observation = self.get_state(observation)
            r.rewards.append(reward)
            r.actions.append(action)
            t += 1

        return r

    def get_discounted_cumulative_reward(self, rewards):
        """
        Parameters
        ----------
        r : list

        Returns
        ----------
        cumulative_reward : list
        """
        cumulative_reward = [0]
        for t, r in enumerate(rewards):
            temp = cumulative_reward[-1] + self._discount ** t * r
            cumulative_reward.append(temp)

        return np.sum(cumulative_reward[1:])

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        integer indicating the action that should be taken
        """
        batch = np.array([state])
        action = self._policy.predict(batch)

        if not env_action_space_is_discrete(self._env):
            return action

        if not greedy:
            return np.random.choice(np.arange(self.num_actions), p=action)

        return self.argmax_break_ties(action)

    def argmax_break_ties(self, probs):
        """
        Breaks ties randomly in an array of probabilities

        Parameters
        ----------
        probs : numpy array, shape = (1, ?)

        Returns
        ----------
        integer indicating the action that should be taken
        """
        return np.random.choice(np.where(probs == probs.max())[0])

    def get_state(self, state):
        """
        Get the state, allows for building state featurizers here
        like tile coding
        """
        return self._state_featurizer(state)
