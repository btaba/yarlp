"""
DDQN
"""
import time
import numpy as np

from yarlp.utils.env_utils import GymEnv
from yarlp.agent.base_agent import Agent
from yarlp.model.networks import cnn
from dateutil.relativedelta import relativedelta as rd
from yarlp.model.model_factories import ddqn_model_factory
from yarlp.model.model_factories import build_ddqn_update_feed_dict
from yarlp.utils.replay_buffer import ReplayBuffer
from yarlp.external.baselines.baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from yarlp.utils.schedules import LinearSchedule
from yarlp.utils import experiment_utils


class DDQNAgent(Agent):
    """
    """

    def __init__(self, env,
                 policy_network=None,
                 policy_network_params={'dueling': True},
                 policy_learning_rate=1e-4,
                 model_file_path=None,
                 buffer_size=10000,
                 exploration_fraction=0.1,
                 exploration_final_eps=0.01,
                 train_freq=4,
                 batch_size=32,
                 target_network_update_freq=1000,
                 prioritized_replay=True,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,
                 max_timesteps=1000000,
                 checkpoint_freq=10000,
                 grad_norm_clipping=10,
                 learning_start_timestep=1000,
                 discount_factor=0.99,
                 double_q=True,
                 reward_len=100,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        # assert env is discrete
        assert GymEnv.env_action_space_is_discrete(env),\
            "env {} is not discrete for DDQNAgent".format(env)
        assert batch_size < buffer_size,\
            'batch_size {} must be less than buffer_size {}'.format(
                batch_size, buffer_size)

        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.checkpoint_freq = checkpoint_freq
        self.max_timesteps = max_timesteps
        self.learning_start_timestep = learning_start_timestep
        self.global_t = 0

        if policy_network is None:
            policy_network = cnn
        elif isinstance(policy_network, str):
            policy_network = experiment_utils.get_network(
                policy_network, {})

        self._policy = ddqn_model_factory(
            env, network=policy_network,
            network_params=policy_network_params,
            learning_rate=policy_learning_rate,
            model_file_path=model_file_path,
            grad_norm_clipping=grad_norm_clipping,
            double_q=double_q, discount_factor=discount_factor)
        self.tf_object_attributes.add('_policy')
        self._policy.G(self._policy['update_target_network'])

        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger.logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        # Create the replay buffer
        self.prioritized_replay = prioritized_replay
        self.buffer_size = buffer_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.set_replay_buffer()

        # Create the schedule for exploration starting from 1.
        schedule_timesteps = max(int(exploration_fraction * max_timesteps), 1)
        exploration = LinearSchedule(
            schedule_timesteps=schedule_timesteps,
            initial_p=1.0,
            final_p=exploration_final_eps)
        self.exploration = exploration

    def set_replay_buffer(self):
        if self.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                self.prioritized_replay_beta_iters = self.max_timesteps
            beta_schedule = LinearSchedule(
                self.prioritized_replay_beta_iters,
                initial_p=self.prioritized_replay_beta0,
                final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(self.buffer_size)
            beta_schedule = None
        self.replay_buffer = replay_buffer
        self.beta_schedule = beta_schedule

    def get_action(self, state, epsilon):
        """
        Generate an epsilon-greedy action from our policy model

        Returns
        ----------
        action : numpy array or integer
        """

        q_values = self._policy.G(
            self._policy['q_output'],
            feed_dict={
                self._policy['state']: state
            })

        deterministic_actions = np.argmax(q_values, axis=1)
        batch_size = q_values.shape[0]
        random_actions = np.random.randint(
            0, q_values.shape[1], batch_size)

        actions = np.where(
            np.random.uniform(size=batch_size) < epsilon,
            random_actions,
            deterministic_actions)

        return actions

    def train(self):

        obs = self._env.reset()
        self.t = 0
        self.last_saved_reward = None
        num_episodes = 0

        while self.global_t < self.max_timesteps:
            epsilon = self.exploration.value(self.global_t)
            action = self.get_action(
                np.expand_dims(self.norm_obs_if_atari(obs), 0),
                epsilon)
            new_obs, reward, done, info = self._env.step(action[0])
            self.replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            if done:
                obs = self._env.reset()

            if self.t > self.learning_start_timestep \
                    and self.t % self.train_freq == 0:
                if self.prioritized_replay:
                    experience = self.replay_buffer.sample(
                        self.batch_size,
                        beta=self.beta_schedule.value(self.global_t))
                    (obst, actions, rewards,
                     obst1, dones, weights, batch_idx) = experience
                else:
                    experience = self.replay_buffer.sample(
                        self.batch_size)
                    obst, actions, rewards, obst1, dones = experience
                    weights, batch_idx = np.ones_like(rewards), None

                obst = self.norm_obs_if_atari(obst)
                obst1 = self.norm_obs_if_atari(obst1)
                args = obst, actions, rewards, obst1, dones, weights
                feed_dict = build_ddqn_update_feed_dict(
                    self._policy, *args)
                self._policy.update(*args)

                if self.prioritized_replay:
                    td_errors = self._policy.G(self._policy['td_errors'], feed_dict)
                    new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(
                        batch_idx, new_priorities)

            if self.t > 0 \
                    and len(info['rewards']) > num_episodes\
                    and self.t > self.learning_start_timestep:
                # log things
                self.logger.set_metrics_for_iter(info['rewards'][num_episodes:])
                num_episodes = len(info['rewards'])
                self.logger.add_metric('timesteps_so_far', self.global_t)
                eta = (self.max_timesteps - self.global_t) / (self.global_t /
                    round(time.time() - self.logger._start_time, 6))
                self.logger.add_metric(
                    'ETA',
                    str(rd(seconds=eta)))
                self.logger.add_metric(
                    'epsilon', self.exploration.value(self.global_t))
                if self.prioritized_replay:
                    self.logger.add_metric(
                        'beta', self.beta_schedule.value(self.global_t))
                self.logger.add_metric('env_id', self._env_id)
                self.logger.add_metric('episodes', num_episodes)
                self.logger.log()

            if self.t > self.learning_start_timestep \
                    and self.t % self.target_network_update_freq == 0:
                self._policy.G(
                    self._policy['update_target_network'])

            running_reward = np.mean(self.logger._running_reward)
            if self.t > self.learning_start_timestep \
                    and self.t % self.checkpoint_freq == 0 \
                    and self.logger._log_dir is not None\
                    and (self.last_saved_reward is None or
                         self.last_saved_reward < running_reward):
                self.logger.logger.info(
                    'Saving model, {} -> {}'.format(
                        self.last_saved_reward, running_reward))
                self.last_saved_reward = running_reward
                self.save(self.logger._log_dir)
            self.t += 1
            self.global_t += 1

        if self.logger._log_dir:
            self.save(self.logger._log_dir, 'final_agent')

    def enjoy(self):

        env = self.env
        done = True
        rewards = 0
        while True:

            if done:
                obs = env.reset()
                print(rewards)
                rewards = 0
            obs = self.norm_obs_if_atari(obs)
            obs = np.expand_dims(obs, 0)
            obs, r, done, _ = env.step(
                self.get_action(obs, epsilon=0.05))
            rewards += r
            env.render()
