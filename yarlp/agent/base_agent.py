"""
    Base Agent class, which takes in a PolicyModel object
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.metric_logger import MetricLogger
from yarlp.utils.replay_buffer import Rollout
from yarlp.utils import tf_utils
from yarlp.model.linear_baseline import LinearFeatureBaseline


ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """

    def __init__(self, env, discount_factor=1,
                 logger=None, seed=None,
                 state_featurizer=lambda x: x):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        if seed:
            tf_utils.set_global_seeds(seed)
            env.seed(seed)
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

        if truncate and steps_sampled > 0 and len(rollouts[-1]) > 1:
            steps_to_remove = steps_sampled - n_steps
            rollouts[-1] = self._truncate_rollout(
                rollouts[-1], steps_to_remove)

        return rollouts

    def _truncate_rollout(self, rollout, steps_to_remove):
        r = Rollout([], [], [], [])
        r.rewards.extend(rollout.rewards[:-steps_to_remove])
        r.actions.extend(rollout.actions[:-steps_to_remove])
        r.states.extend(rollout.states[:-steps_to_remove])
        r.done.extend(rollout.done[:-steps_to_remove])
        return r

    def rollout(self, render=False, render_freq=5, greedy=False):
        """
        Performs actions on the environment
        based on the agent's current weights for 1 single rollout

        render: bool, whether to render episodes in a video

        Returns
        ----------
        Rollout : named tuple
        """
        r = Rollout([], [], [], [])

        observation = self._env.reset()
        observation = self.get_state(observation)
        r.done.append(0)
        for t in range(self._env.spec.timestep_limit):
            r.states.append(observation)
            action = self.get_action(observation, greedy=greedy)
            (observation, reward, done, _) = self._env.step(action)

            if render and t and t % render_freq == 0:
                self._env.render()

            observation = self.get_state(observation)
            r.done.append(done)
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

    def get_discounted_reward_list(self, rewards, discount=None):
        """
        Given a list of rewards, return the discounted rewards
        at each time step, in linear time
        """
        if discount is None:
            discount = self._discount

        rt = 0
        discounted_rewards = []
        for t in range(len(rewards) - 1, -1, -1):
            rt = rewards[t] + discount * rt
            discounted_rewards.append(rt)

        return list(reversed(discounted_rewards))

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        action : numpy array or integer
        """
        batch = np.array([state])
        a = self._policy.policy.predict(
            self._policy.get_session(),
            batch, greedy)[0]
        return a

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


class BatchAgent(Agent):
    """
    Abstract class for an agent.
    """

    def __init__(self, *args, **kwargs):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update(self, path):
        """
        Parameters
        ----------
        path : dict
        """
        pass

    def train(self, num_train_steps=10, num_test_steps=0,
              n_steps=1024, max_timesteps=0,
              render=False,
              whiten_advantages=True,
              truncate_rollouts=False):
        """
        Parameters
        ----------
        num_train_steps : integer
            Total number of training iterations.

        num_test_steps : integer
            Number of testing iterations per training iteration.

        n_steps : integer
            Total number of samples from the environment for each
            training iteration.

        max_timesteps : integer
            maximum number of total steps to execute in the environment

        whiten_advantages : bool, whether to whiten the advantages

        render : bool, whether to render episodes in a video

        Returns
        ----------
        None
        """
        assert sum([num_train_steps > 0,
                    max_timesteps > 0]) == 1,\
            "Must provide at least one limit to training"

        timesteps_so_far = 0
        train_steps_so_far = 0

        while True:

            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif num_train_steps and train_steps_so_far >= num_train_steps:
                break

            # execute an episode
            rollouts = self.rollout_n_steps(
                n_steps, render=render, truncate=truncate_rollouts)

            actions = []
            states = []
            advantages = []
            td_returns = []

            for rollout in rollouts:

                baseline_pred = np.zeros((len(rollout.rewards)))
                if self._baseline_model:
                    baseline_pred = self._baseline_model.predict(
                        np.array(rollout.states)).flatten()

                is_terminal = rollout.done[-1] == 1
                if not is_terminal:
                    # the episode did not terminate,
                    # so we truncate the last step so that we can use
                    # baseline_pred[-1] as the discounted future reward
                    rollout = self._truncate_rollout(rollout, 1)
                else:
                    # the episode terminated, so the future reward is 0
                    baseline_pred = np.append(baseline_pred, 0)

                advantage = rollout.rewards + self._discount *\
                    baseline_pred[1:] - baseline_pred[:-1]
                advantage = self.get_discounted_reward_list(
                    advantage, discount=self._discount * self._gae_lambda)

                advantages = np.concatenate([advantages, advantage])
                states.append(rollout.states)
                actions.append(rollout.actions)
                td_returns = np.concatenate(
                    [td_returns, baseline_pred[:-1] + advantage])

            states = np.concatenate([s for s in states])
            actions = np.concatenate([a for a in actions])

            if whiten_advantages:
                advantages = (advantages - np.mean(advantages)) /\
                    (np.std(advantages) + 1e-8)

            # batch update the baseline model
            if isinstance(self._baseline_model, LinearFeatureBaseline):
                self._baseline_model.fit(states, td_returns)
            elif hasattr(self._baseline_model, 'G'):
                self._baseline_model.update(
                    states, td_returns)

            # update the policy
            path_dict = {
                'states': states,
                'actions': actions,
                'td_returns': td_returns,
                'advantages': advantages
            }
            self.update(path_dict)

            if not is_terminal:
                rollouts = rollouts[:-1]
            self.logger.set_metrics_for_rollout(rollouts, train=True)
            self.logger.log()

            if num_test_steps > 0:
                r = []
                for t_test in range(num_test_steps):
                    rollout = self.rollout(greedy=True)
                    r.append(rollout)
                self.logger.add_metric('policy_loss', 0)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()

            if self.logger._log_dir is not None:
                self.save_models(self.logger._log_dir)

            timesteps_so_far += advantages.shape[0]
            train_steps_so_far += 1

        return
