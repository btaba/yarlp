"""
DQN and DDQN
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
from yarlp.utils.schedules import LinearSchedule, PiecewiseSchedule, ConstantSchedule
from yarlp.utils import experiment_utils
from yarlp.utils.env_utils import get_wrapper_by_name


class DDQNAgent(Agent):
    """
    """

    def __init__(self, env,
                 policy_network=None,
                 policy_network_params={'dueling': False},
                 learning_rate_schedule=None,
                 policy_learning_rate=1e-4,
                 model_file_path=None,
                 buffer_size=10000,
                 exploration_fraction=0.1,
                 exploration_final_eps=0.01,
                 exploration_schedule=None,
                 train_freq=4,
                 batch_size=32,
                 target_network_update_freq=10000,
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,
                 max_timesteps=1000000,
                 checkpoint_freq=10000,
                 save_freq=20000,
                 grad_norm_clipping=10,
                 learning_start_timestep=10000,
                 discount_factor=0.99,
                 double_q=True,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        # assert env is discrete
        assert GymEnv.env_action_space_is_discrete(env),\
            "env {} is not discrete for DDQNAgent".format(env)
        assert batch_size < buffer_size,\
            'batch_size {} must be less than buffer_size {}'.format(
                batch_size, buffer_size)
        prioritized_replay = False
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_freq = save_freq
        self.max_timesteps = max_timesteps
        self.learning_start_timestep = learning_start_timestep
        self.learning_rate = policy_learning_rate
        self.global_t = 0

        if policy_network is None:
            policy_network = cnn
        elif isinstance(policy_network, str):
            policy_network = experiment_utils.get_network(
                policy_network, {})

        self._policy = ddqn_model_factory(
            env, network=policy_network,
            network_params={},
            model_file_path=None,
            grad_norm_clipping=10,
            double_q=False, discount_factor=0.99)
        self.tf_object_attributes.add('_policy')
        self._policy.G(self._policy['update_target_network'])

        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger.logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        # Create the replay buffer
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.buffer_size = buffer_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.set_replay_buffer()

        # Create exploration and learning rate schedules
        if exploration_schedule is None:
            schedule_timesteps = max(
                int(exploration_fraction * max_timesteps), 1)
            exploration = LinearSchedule(
                schedule_timesteps=schedule_timesteps,
                initial_p=1.0,
                final_p=exploration_final_eps)
        elif isinstance(exploration_schedule, list):
            exploration = PiecewiseSchedule(
                exploration_schedule,
                outside_value=exploration_schedule[-1][-1]
            )
        self.exploration = exploration

        if learning_rate_schedule is None:
            lr = ConstantSchedule(policy_learning_rate)
        elif isinstance(learning_rate_schedule, list):
            lr = PiecewiseSchedule(
                learning_rate_schedule,
                outside_value=learning_rate_schedule[-1][-1])
        self.lr_schedule = lr

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
        self.last_saved_reward = -np.inf
        update_target_network = self._policy['update_target_network']
        update_fn = self._policy['optimizer_op:']

        while self.global_t < self.max_timesteps:

            if self.t < self.learning_start_timestep:
                epsilon = 1
            else:
                epsilon = self.exploration.value(self.global_t)

            action = self.get_action(
                np.expand_dims(np.array(obs), 0),
                epsilon)

            new_obs, reward, done, info = self._env.step(action[0])
            self.replay_buffer.add(obs, action, reward, new_obs, done)
            obs = new_obs

            if done:
                obs = self._env.reset()

            # train the networks every train_freq
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

                obst = np.array(obst)
                obst1 = np.array(obst1)
                lr = self.lr_schedule.value(self.global_t)

                args = obst, actions, rewards, obst1, dones, weights, lr
                feed_dict = build_ddqn_update_feed_dict(
                    self._policy, *args)
                self._policy.G(update_fn, feed_dict)

                if self.prioritized_replay:
                    td_errors = self._policy.G(
                        self._policy['td_errors'], feed_dict)
                    new_priorities = np.abs(td_errors) + \
                        self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(
                        batch_idx, new_priorities)

            # log stuff every episode
            if self.t > 0 \
                    and self.t % self.checkpoint_freq == 0\
                    and self.t > self.learning_start_timestep:
                # log things
                episode_rewards = get_wrapper_by_name(
                    self._env, "MonitorEnv").get_episode_rewards()
                self.logger.set_metrics_for_iter(episode_rewards[-100:])
                num_episodes = len(episode_rewards)
                self.logger.add_metric('timesteps_so_far', self.global_t)
                eta = (self.max_timesteps - self.t) /\
                    (self.t / round(time.time() - self.logger._start_time, 6))
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
                self.logger.add_metric(
                    'last_saved_reward', self.last_saved_reward)
                self.logger.add_metric(
                    'learning_rate', self.lr_schedule.value(self.global_t))
                self.logger.log()

            if self.t > self.learning_start_timestep \
                    and self.t % self.target_network_update_freq == 0:
                self.logger.logger.info('Update target network')
                self._policy.G(update_target_network)

            # save model if necessary
            if self.t > self.learning_start_timestep \
                    and self.t % self.save_freq == 0 \
                    and self.logger._log_dir is not None:

                running_reward = np.mean(get_wrapper_by_name(
                    self._env, "MonitorEnv").get_episode_rewards()[-100:])

                if running_reward > self.last_saved_reward:
                    self.logger.logger.info(
                        'Saving best model, {} -> {}'.format(
                            self.last_saved_reward, running_reward))
                    self.save(self.logger._log_dir, 'best_agent')
                    self.last_saved_reward = running_reward

                self.logger.logger.info('Saving model for checkpoint')
                self.save(self.logger._log_dir)
            self.t += 1
            self.global_t += 1

        if self.logger._log_dir:
            self.save(self.logger._log_dir)

    def enjoy(self, t):
        from time import sleep
        env = self.env
        done = True
        rewards = 0
        while True:

            if done:
                obs = env.reset()
                print(rewards, 'done')
                rewards = 0
            sleep(t)
            obs = np.expand_dims(obs, 0)
            obs, r, done, _ = env.step(
                self.get_action(obs, epsilon=0.05))
            rewards += r
            env.render()
