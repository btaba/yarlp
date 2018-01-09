"""
DDQN
"""

import numpy as np

from collections import deque
from yarlp.utils.env_utils import GymEnv
from yarlp.agent.base_agent import Agent
from yarlp.utils.experiment_utils import get_network
from yarlp.model.networks import cnn
from yarlp.model.model_factories import ddqn_model_factory
from yarlp.model.model_factories import build_ddqn_update_feed_dict
from yarlp.external.baselines.baselines.deepq.replay_buffer import ReplayBuffer
from yarlp.external.baselines.baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from yarlp.external.baselines.baselines.common.schedules import LinearSchedule


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
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,
                 max_timesteps=10000000,
                 checkpoint_freq=10000,
                 grad_norm_clipping=10,
                 # log_freq=2000,
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

        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.checkpoint_freq = checkpoint_freq
        self.max_timesteps = max_timesteps
        self.learning_start_timestep = learning_start_timestep
        # self.log_freq = log_freq

        if policy_network is None:
            policy_network = cnn

        self._policy = ddqn_model_factory(
            env, network=policy_network,
            network_params=policy_network_params,
            learning_rate=policy_learning_rate,
            model_file_path=model_file_path,
            grad_norm_clipping=grad_norm_clipping,
            double_q=double_q)
        self.tf_object_attributes.add('_policy')
        self._policy.G(self._policy['update_target_network'])
        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger.logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(
                buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            beta_schedule = LinearSchedule(
                prioritized_replay_beta_iters,
                initial_p=prioritized_replay_beta0,
                final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        self.replay_buffer = replay_buffer
        self.beta_schedule = beta_schedule

        # Create the schedule for exploration starting from 1.
        schedule_timesteps = max(int(exploration_fraction * max_timesteps), 1)
        exploration = LinearSchedule(
            schedule_timesteps=schedule_timesteps,
            initial_p=1.0,
            final_p=exploration_final_eps)
        self.exploration = exploration

    def get_action(self, state, epsilon, greedy=False):
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
        episode_returns = []
        total_reward = 0
        last_saved_reward = None
        while self.t < self.max_timesteps:
            epsilon = self.exploration.value(self.t)
            action = self.get_action(np.expand_dims(obs, 0), epsilon)
            new_obs, reward, done, _ = self._env.step(action[0])
            self.replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs
            total_reward += reward

            if done:
                obs = self._env.reset()
                episode_returns.append(total_reward)
                total_reward = 0

            if self.t > self.learning_start_timestep \
                    and self.t % self.train_freq == 0:
                if self.prioritized_replay:
                    experience = self.replay_buffer.sample(
                        self.batch_size,
                        beta=self.beta_schedule.value(self.t))
                    (obst, actions, rewards,
                     obst1, dones, weights, batch_idx) = experience
                else:
                    experience = self.replay_buffer.sample(
                        self.batch_size)
                    obst, actions, rewards, obst1, dones = experience
                    weights, batch_idx = np.ones_like(rewards), None

                # actions = np.squeeze(actions)
                args = obst, actions, rewards, obst1, dones, weights
                # for a in args:
                #     print(a.shape)
                feed_dict = build_ddqn_update_feed_dict(
                    self._policy, *args)
                # loss_before = self._policy.G(self._policy['loss'], feed_dict)
                self._policy.update(*args)
                # loss_after = self._policy.G(self._policy['loss'], feed_dict)

                # self.logger.add_metric('LossBefore', loss_before)
                # self.logger.add_metric('LossAfter', loss_after)
                # self.logger.log()

                if self.prioritized_replay:
                    td_errors = self._policy.G(self._policy['td_errors'])
                    new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(
                        batch_idx, new_priorities)

            if self.t > 0 \
                    and len(episode_returns) > 10\
                    and self.t > self.learning_start_timestep:
                self.logger.set_metrics_for_iter(episode_returns)
                self.logger.add_metric('timesteps_so_far', self.t)
                self.logger.add_metric(
                    'epsilon', self.exploration.value(self.t))
                self.logger.add_metric('env_id', self._env_id)
                self.logger.log()
                episode_returns = []

            if self.t > self.learning_start_timestep \
                    and self.t % self.target_network_update_freq == 0:
                self._policy.G(
                    self._policy['update_target_network'])
            running_reward = self.logger._running_reward
            if self.t > self.learning_start_timestep \
                    and self.t % self.checkpoint_freq == 0 \
                    and self.logger._log_dir is not None\
                    and (last_saved_reward is None or
                         last_saved_reward > running_reward):
                self.logger.logger.info(
                    'Saving model, {} -> {}'.format(
                        last_saved_reward, running_reward))
                self.save(self.logger._log_dir)
            self.t += 1
