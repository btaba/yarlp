"""
Random actions as a baseline
"""

from yarlp.agent.base_agent import Agent
from yarlp.utils.logger import logger

import numpy as np


class RandomAgent(Agent):
    """Agent that takes random actions
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self._policy = self

    def train(self, num_train_steps, num_test_steps=0):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        Returns
        ----------
        """
        for i in range(num_train_steps):
            # execute an episode
            rollout = self.rollout()

            for t_test in range(num_test_steps):
                self.do_greedy_episode()
                # gather stats

            logger.info('Training Step {}'.format(i))
            logger.info('Episode length {}'.format(len(rollout.rewards)))
            logger.info('Average reward {}'.format(np.mean(rollout.rewards)))
            logger.info('Std reward {}'.format(np.std(rollout.rewards)))
            logger.info('Total reward {}'.format(np.sum(rollout.rewards)))

        return

    def get_action(self, states):
        if len(states.shape) == 1:
            return self._env.action_space.sample()

        return np.array(
            [self._env.action_space.sample() for _ in range(states.shape[0])])
