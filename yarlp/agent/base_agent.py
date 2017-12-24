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
from yarlp.model.networks import normc_initializer, mlp
from yarlp.utils.experiment_utils import get_network
from yarlp.model.model_factories import value_function_model_factory

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
        if not hasattr(self, '_baseline_model'):
            return np.zeros(len(obs))

        if self._baseline_model is None:
            return np.zeros(len(obs))

        return self._baseline_model.predict(
            np.array(obs)).flatten()

    def get_action(self, state, greedy=False):
        """
        Generate an action from our policy model

        Returns
        ----------
        action : numpy array or integer
        """
        batch = np.array([state])
        with self._policy.G._session.as_default():
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


def do_rollout(agent, env, n_steps=None,
               render=False, render_freq=5, greedy=False):
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
    observation = agent.get_state(observation)
    done = False

    observations = []
    rewards = []
    dones = []
    actions = []

    while True:

        if render and t and t % render_freq == 0:
            env.render()

        is_truncated_rollout = n_steps is not None and t > 0 \
            and t % n_steps == 0
        is_completed_rollout = n_steps is None and done is True

        if done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_return = 0
            episode_length = 0
            observation = env.reset()
            done = False

        if is_truncated_rollout or is_completed_rollout:

            next_baseline_pred = agent.get_baseline_pred(
                [observation]) * (1 - dones[-1])
            baseline_preds = agent.get_baseline_pred(np.array(observations))

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
            dones = []
            actions = []
            t = 0

        action = agent.get_action(observation, greedy=greedy)
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
    Abstract class for a batch agent.
    """

    def __init__(self, env,
                 baseline_network=None,
                 baseline_model_learning_rate=1e-2,
                 baseline_train_iters=3,
                 baseline_network_params={'final_weights_initializer': normc_initializer(1.0)},
                 *args, **kwargs):
        """
        """
        super().__init__(env, *args, **kwargs)

        self.baseline_train_iters = baseline_train_iters
        if isinstance(baseline_network, LinearFeatureBaseline):
            self._baseline_model = baseline_network
        elif baseline_network is None:
            self._baseline_model = LinearFeatureBaseline()
        else:
            baseline_network = get_network(baseline_network, baseline_network_params)
            self._baseline_model = value_function_model_factory(
                env, baseline_network,
                learning_rate=baseline_model_learning_rate)

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
              *args, **kwargs):
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

        rollout_gen = do_rollout(
            self, self._env, n_steps, greedy=False,
            *args, **kwargs)

        train_steps_so_far = 0
        timesteps_so_far = 0
        assert sum([max_timesteps > 0, num_train_steps > 0]) == 1,\
            "Either max_timesteps > 0 or num_train_steps > 0"

        while True:
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif num_train_steps and train_steps_so_far >= num_train_steps:
                break

            rollout = rollout_gen.__next__()
            train_steps_so_far += 1
            timesteps_so_far += len(rollout['dones'])

            add_advantage(rollout, self._discount, self._gae_lambda)

            adv = rollout['advantages']
            if whiten_advantages:
                adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
                rollout['advantages'] = adv

            self.update(rollout)

            # batch update the baseline model
            if isinstance(self._baseline_model, LinearFeatureBaseline):
                self._baseline_model.fit(
                    rollout['observations'], rollout['discounted_future_reward'])
            elif hasattr(self._baseline_model, 'G'):
                data = [rollout['observations'], rollout['discounted_future_reward']]
                self.logger.add_metric(
                    'Baseline_Loss_Before',
                    self._baseline_model.eval_tensor(self._baseline_model.loss, *data))
                for _ in range(self.baseline_train_iters):
                    for ob, a in tf_utils.iterbatches(
                            [rollout['observations'], rollout['discounted_future_reward']]):
                        self._baseline_model.update(ob, a)
                self.logger.add_metric(
                    'Baseline_Loss_After',
                    self._baseline_model.eval_tensor(self._baseline_model.loss, *data))

            self.logger.add_metric('timesteps_so_far', timesteps_so_far)
            self.logger.add_metric('env_id', self._env_id)
            self.logger.set_metrics_for_rollout(rollout, train=True)
            self.logger.log()

            if num_test_steps > 0:
                test_gen = do_rollout(
                    self, self._env, greedy=True)
                r = []
                for t_test in range(num_test_steps):
                    rollout = test_gen.__next__()
                    r.append(rollout)
                self.logger.set_metrics_for_rollout(r, train=False)
                self.logger.log()
