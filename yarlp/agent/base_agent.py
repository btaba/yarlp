"""
    Base Agent class, which takes in a PolicyModel object
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from yarlp.utils.env_utils import GymEnv
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

    def save_models(self, path):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        pass

    @property
    def env(self):
        return self._env

    @property
    def num_actions(self):
        return GymEnv.get_env_action_space_dim(self._env)

    def rollout_n_steps(self, n_steps=1000, truncate=False, **kwargs):
        """
        Do rollouts until we have have achieved `n_steps` steps in the env.

        Parameters
        ----------
        n_steps : int, the number of steps to sample in the environment
        truncate : bool, whether to truncate the last episode to the
            exact number of steps specified in `n_steps`. If False, we could
            have more steps than `n_steps` sampled.

        Returns
        ----------
        List(Rollout) : list of Rollout
        """

        steps_sampled = 0
        rollouts = []
        while steps_sampled < n_steps:
            r = self.rollout(**kwargs)
            steps_sampled += len(r.rewards)
            rollouts.append(r)

        if truncate and steps_sampled > 0:
            steps_to_remove = steps_sampled - n_steps
            r = Rollout([], [], [])
            r.rewards.extend(rollouts[-1].rewards[:-steps_to_remove])
            r.actions.extend(rollouts[-1].actions[:-steps_to_remove])
            r.states.extend(rollouts[-1].states[:-steps_to_remove])
            rollouts[-1] = r

        return rollouts

    def rollout(self, render=False, render_freq=5, greedy=False):
        """
        Performs actions on the environment
        based on the agent's current weights

        render: bool, whether to render episodes in a video

        Returns
        ----------
        Rollout : named tuple
        """
        r = Rollout([], [], [])

        observation = self._env.reset()
        observation = self.get_state(observation)
        for t in range(self._env.spec.timestep_limit):
            r.states.append(observation)
            action = self.get_action(observation, greedy=greedy)
            (observation, reward, done, _) = self._env.step(action)

            if render and t and t % render_freq == 0:
                self._env.render()

            observation = self.get_state(observation)
            r.rewards.append(reward)
            r.actions.append(action)
            if done:
                break

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

    def get_discounted_reward_list(self, rewards):
        """
        Given a list of rewards, return the discounted rewards
        at each time step, in linear time
        """
        rt = 0
        discounted_rewards = []
        for t in range(len(rewards) - 1, -1, -1):
            rt = rewards[t] + self._discount * rt
            discounted_rewards.append(rt)

        return list(reversed(discounted_rewards))

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        integer indicating the action that should be taken
        """
        batch = np.array([state])

        if not GymEnv.env_action_space_is_discrete(self._env):
            if greedy:
                return self._policy.predict(batch, output_name='output:greedy')
            return self._policy.predict(batch)

        action = self._policy.predict(batch)
        if greedy:
            return self.argmax_break_ties(action)

        return np.random.choice(np.arange(self.num_actions), p=action)

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
