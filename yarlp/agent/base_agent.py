"""
    Base Agent class, which takes in a PolicyModel object
"""

import gym
import numpy as np
from abc import ABCMeta, abstractmethod
from yarlp.utils.env_utils import GymEnv
from yarlp.utils.metric_logger import MetricLogger
from yarlp.utils import tf_utils
from yarlp.model.linear_baseline import LinearFeatureBaseline

ABC = ABCMeta('ABC', (object,), {})


class Agent(ABC):
    """
    Abstract class for an agent.
    """

    def __init__(self, env, discount_factor=0.99,
                 logger=None, seed=None, gae_lambda=0,
                 state_featurizer=lambda x: x):
        """
        discount_factor : float
            Discount rewards by this factor
        """
        # Discount factor
        assert discount_factor >= 0 and discount_factor <= 1
        self._discount = discount_factor
        self.gae_lambda = 0

        if logger is None:
            self.logger = MetricLogger()
        else:
            self.logger = logger

        if seed is not None:
            self.logger._logger.info('Seed: {}'.format(seed))
            tf_utils.set_global_seeds(seed)
            env.seed(seed)
        self._env = env
        self._env_id = '{}_gym{}'.format(
            env.spec.id, gym.__version__)

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

    def get_baseline_pred(self, obs):
        if self._baseline_model:
            return self._baseline_model.predict(
                np.array([obs])).flatten()[0]
        return None

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
        with self._policy.G._session.as_default():
            # a = self._policy.policy.predict(
            #     # self._policy.get_session(),
            #     batch[0], greedy)
            a, _ = self._policy.pi.act(not greedy, batch[0])
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


def do_rollout(agent, env, n_steps=None, render=False, render_freq=5,
               greedy=False):
    """
    Performs actions on the environment
    based on the agent's current weights for 1 single rollout

    render: bool, whether to render episodes in a video

    Returns
    ----------
    dict
    """
    t = 0
    episode_return = 0
    episode_length = 0
    episode_returns = []
    episode_lengths = []

    observation = env.reset()
    # observation = agent.get_state(observation)
    done = False

    observations = []
    rewards = []
    baseline_preds = []
    dones = []
    actions = []

    while True:

        if render and t and t % render_freq == 0:
            env.render()

        if done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_return = 0
            episode_length = 0
            observation = env.reset()
            done = False

        is_truncated_rollout = n_steps is not None and t > 0 \
            and t % n_steps == 0
        is_completed_rollout = n_steps is None and done is True

        if is_truncated_rollout or is_completed_rollout:

            # next_baseline_pred = agent.get_baseline_pred(
            #     observation) * (1 - dones[-1])
            next_baseline_pred = agent.act(not greedy, observation)[1] *\
                (1 - dones[-1])

            rollout = {
                "observations": np.array(observations),
                "rewards": np.array(rewards),
                "baseline_preds": baseline_preds,
                "dones": dones,
                "actions": np.array(actions),
                "next_baseline_pred": next_baseline_pred,
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths
            }

            yield rollout

            episode_returns = []
            episode_lengths = []
            observations = []
            rewards = []
            baseline_preds = []
            dones = []
            actions = []
            t = 0

        # action = agent.get_action(observation, greedy=greedy)
        action, baseline_pred = agent.act(not greedy, observation)
        # baseline_pred = agent.get_baseline_pred(observation)

        baseline_preds.append(baseline_pred)
        actions.append(action)
        observations.append(observation)

        observation, reward, done, _ = env.step(action)
        dones.append(done)
        rewards.append(reward)

        episode_return += reward
        episode_length += 1
        t += 1


def add_advantage(rollout, gamma, Lambda=0):
    baseline_preds = np.append(
        rollout["baseline_preds"], rollout["next_baseline_pred"])
    T = len(rollout["rewards"])
    rollout["advantages"] = gaelam = np.empty(T, 'float32')
    rewards = rollout["rewards"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - rollout['dones'][t]
        delta = rewards[t] + gamma * baseline_preds[t + 1] * nonterminal -\
            baseline_preds[t]
        gaelam[t] = lastgaelam = delta + \
            gamma * Lambda * nonterminal * lastgaelam
    rollout["discounted_future_reward"] = rollout["advantages"] +\
        rollout["baseline_preds"]


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

    def train(self, num_train_steps=0, num_test_steps=0,
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
                    max_timesteps > 0]) >= 1,\
            "Must provide num_train_steps or max_timesteps > 0."

        timesteps_so_far = 0
        train_steps_so_far = 0

        rollout_gen = do_rollout(
            self._policy, self._env, n_steps, greedy=False)

        while True:

            if num_train_steps and train_steps_so_far >= num_train_steps:
                break
            elif max_timesteps and timesteps_so_far >= max_timesteps:
                break

            rollout = rollout_gen.__next__()
            add_advantage(rollout, self._discount, self.gae_lambda)

            adv = rollout['advantages']
            if whiten_advantages:
                adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
                rollout['advantages'] = adv

            # # batch update the baseline model
            # # if isinstance(self._baseline_model, LinearFeatureBaseline):
            # #     self._baseline_model.fit(rollout['observations'], adv)
            # # elif hasattr(self._baseline_model, 'G'):
            # #     self._baseline_model.update(
            # #         rollout['observations'], adv)

            self.update(rollout)

            timesteps_so_far += adv.shape[0]
            train_steps_so_far += 1

            self.logger.add_metric('timesteps_so_far', timesteps_so_far)
            self.logger.add_metric('env_id', self._env_id)
            self.logger.set_metrics_for_rollout(rollout, train=True)
            self.logger.log()

            if num_test_steps > 0:
                test_gen = do_rollout(
                    self.policy, self._env, greedy=True)
                r = []
                for t_test in range(num_test_steps):
                    rollout = test_gen.__next__()
                    r.append(rollout)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()
