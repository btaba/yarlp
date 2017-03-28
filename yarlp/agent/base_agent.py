"""
    Base Agent class, which takes in a PolicyModel object
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple


ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """

    def __init__(self, env, num_max_rollout_steps, discount_factor=1):
        """
        num_max_rollout_steps : integer
            Maximum number of steps executed in an episode

        discount_factor : float
            Discount rewards by this factor
        """
        self._env = env

        # Maximum number of steps in an episode
        self.num_max_rollout_steps = num_max_rollout_steps

        # Discount factor
        assert discount_factor >= 0 and discount_factor <= 1
        self._discount = discount_factor

    @abstractmethod
    def train(self):
        pass

    @property
    def env(self):
        return self._env

    @property
    def num_actions(self):
        if self.env_is_discrete():
            return self._env.action_space.n
        return self._env.action_space.shape[0]

    def env_is_discrete(self):
        if hasattr(self._env.action_space, 'n'):
            return True
        return False

    def rollout(self):
        """
        Performs actions for num_max_rollout_steps on the environment
        based on the agent's current weights

        Returns
        ----------
        Rollout : named tuple
        """

        Rollout = namedtuple('Rollout', 'rewards actions states')
        r = Rollout([], [], [])

        observation = self._env.reset()
        observation = self.get_state(observation)
        for t in range(self.num_max_rollout_steps):
            r.states.append(observation)
            action = self.get_action(observation)
            (observation, reward, done, _) = self._env.step(action)
            observation = self.get_state(observation)
            r.rewards.append(reward)
            r.actions.append(action)
            if done:
                break

        return r

    def do_greedy_episode(self, max_time_steps=1000):
        t = 0
        done = False
        total_reward = 0
        observation = self._env.reset()
        observation = self.get_state(observation)
        while not done and t < max_time_steps:
            action = self.get_action(observation, greedy=True)
            (observation, reward, done, _) = self._env.step(action)
            observation = self.get_state(observation)
            total_reward += reward
            t += 1
        return total_reward

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

        if not self.env_is_discrete():
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
        return state
