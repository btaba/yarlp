"""
Random actions as a baseline
"""

from yarlp.agent.base_agent import Agent

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

            rollout = self.rollout()

            self.logger.set_metrics_for_rollout(rollout, train=True)
            self.logger.log()

            if num_test_steps > 0:
                r = []
                for t_test in range(num_test_steps):
                    rollout = self.rollout(greedy=True)
                    r.append(rollout)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()

        return

    def get_action(self, states, *args, **kwargs):
        if len(states.shape) == 1:
            return self._env.action_space.sample()

        return np.array(
            [self._env.action_space.sample() for _ in range(states.shape[0])])
