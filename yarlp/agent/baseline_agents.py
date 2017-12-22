"""
Random actions as a baseline
"""

from yarlp.agent.base_agent import Agent
from yarlp.agent.base_agent import do_rollout

import numpy as np


class RandomAgent(Agent):
    """Agent that takes random actions
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self._policy = self

    def train(self, num_train_steps=1, n_steps=None,
              num_test_steps=0,
              max_timesteps=0):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        num_test_steps : integer
            Number of testing iterations per training iteration.

        max_timesteps : integer
            maximum number of total steps to execute in the environment

        Returns
        ----------
        """
        assert sum([num_train_steps > 0,
                    max_timesteps > 0]) == 1,\
            "Must provide at least one limit to training"

        timesteps_so_far = 0
        train_steps_so_far = 0

        rollout_gen = do_rollout(
            self, self._env, n_steps, greedy=False)

        while True:

            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif num_train_steps and train_steps_so_far >= num_train_steps:
                break

            rollout = rollout_gen.__next__()

            self.logger.set_metrics_for_rollout(rollout, train=True)
            self.logger.log()

            if num_test_steps > 0:
                r = []
                for t_test in range(num_test_steps):
                    rollout = self.rollout(greedy=True)
                    r.append(rollout)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()

            timesteps_so_far += len(rollout)
            train_steps_so_far += 1

        return

    def get_action(self, states, *args, **kwargs):
        if len(states.shape) == 1:
            return self._env.action_space.sample()

        return np.array(
            [self._env.action_space.sample() for _ in range(states.shape[0])])
