"""
A2C - synchronous advantage actor critic with generalized advantage estimation

Adapted from https://arxiv.org/pdf/1602.01783.pdf, Algorithm S3

"""
import time
import numpy as np
from yarlp.agent.base_agent import Agent, add_advantage
from yarlp.model.networks import cnn
from yarlp.utils.experiment_utils import get_network
from yarlp.model.model_factories import a2c_model_factory
from yarlp.utils.env_utils import ParallelEnvs
from yarlp.utils.metric_logger import explained_variance
from yarlp.utils.schedules import PiecewiseSchedule, ConstantSchedule
from dateutil.relativedelta import relativedelta as rd


class A2CAgent(Agent):

    def __init__(
            self, env,
            policy_network=None,
            policy_network_params={
                'final_dense_weights_initializer': 0.01
            },
            policy_learning_rate=5e-4,
            value_network_params={
                'final_dense_weights_initializer': 1.0
            },
            entropy_weight=0.01,
            model_file_path=None,
            adaptive_std=False, init_std=1.0, min_std=1e-6,
            n_steps=5,
            max_timesteps=1000000,
            grad_norm_clipping=0.5,
            gae_lambda=0.98,
            checkpoint_freq=10000,
            save_freq=50000,
            policy_learning_rate_schedule=None,
            *args, **kwargs):

        super().__init__(env, *args, **kwargs)

        assert isinstance(self._env, ParallelEnvs),\
            "env must be ParallelEnvs class for A2C agent"

        if policy_network is None:
            policy_network = cnn

        pn = get_network(policy_network, policy_network_params)

        self._policy = a2c_model_factory(
            env, network=pn,
            policy_network_params=policy_network_params,
            value_network_params=value_network_params,
            learning_rate=policy_learning_rate,
            has_learning_rate_schedule=True,
            entropy_weight=entropy_weight,
            min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
            grad_norm_clipping=grad_norm_clipping,
            model_file_path=model_file_path)

        self.tf_object_attributes.add('_policy')
        self.unserializables.add('_env')

        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger.logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        self.n_steps = n_steps
        self.max_timesteps = max_timesteps
        self.t = 0
        self.checkpoint_freq = checkpoint_freq
        self.save_freq = save_freq
        self.num_envs = self.env.num_envs
        self.env_id = self.env.env_id
        self.start_seed = self.env.start_seed
        self.is_atari = self.env.is_atari

        if policy_learning_rate_schedule is None:
            lr = ConstantSchedule(policy_learning_rate)
        elif isinstance(policy_learning_rate_schedule, list):
            lr = PiecewiseSchedule(
                policy_learning_rate_schedule,
                outside_value=policy_learning_rate_schedule[-1][-1])
        self.lr_schedule = lr

    def set_env(self):
        self._env = ParallelEnvs(self.env_id, self.num_envs,
                                 self.start_seed, self.is_atari)

    def train(self):

        self.last_saved_reward = -np.inf
        num_envs = self._env.num_envs
        batch_ob_shape = (
            num_envs * self.n_steps, *self.env.observation_space.shape)
        obs = self.env.reset()

        while self.t < self.max_timesteps:
            mb_obs, mb_rewards, mb_actions = [], [], []
            mb_values, mb_dones = [], []

            for n in range(self.n_steps):

                actions = self.get_batch_actions(obs)
                values = self._policy.G(
                    self._policy['vf'],
                    {self._policy['state']: obs})

                mb_obs.append(obs)
                mb_actions.append(actions)
                mb_values.append(values)
                obs, rewards, dones, _ = self.env.step(actions)
                mb_dones.append(dones)
                mb_rewards.append(rewards)
                self.t += num_envs

            mb_obs = np.asarray(mb_obs).swapaxes(1, 0).reshape(batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
            mb_values = np.asarray(mb_values).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones).swapaxes(1, 0)
            last_values = self._policy.G(
                self._policy['vf'],
                {self._policy['state']: obs})

            mb_discounted_rewards = []
            for n in range(self._env.num_envs):
                rollout = {
                    "baseline_preds": mb_values[n].flatten(),
                    "next_baseline_pred": last_values[n].flatten(),
                    "dones": mb_dones[n],
                    "rewards": mb_rewards[n]
                }
                add_advantage(rollout, self._discount, Lambda=0)
                mb_discounted_rewards.append(rollout["discounted_rewards"])

            mb_actions = mb_actions.flatten()
            mb_values = mb_values.flatten()
            mb_discounted_rewards = np.asarray(mb_discounted_rewards).flatten()
            mb_rewards = mb_discounted_rewards

            lr = self.lr_schedule.value(self.t)

            # fit the value and policy networks
            policy_loss = self._policy.update(
                mb_obs, mb_rewards,
                mb_actions, mb_rewards - mb_values, lr)
            policy_update_feed_dict = self._policy.build_update_feed_dict(
                self._policy, mb_obs, mb_rewards,
                mb_actions, mb_rewards - mb_values, lr)
            policy_entropy = self._policy.G(
                self._policy['entropy'], policy_update_feed_dict)
            vf_loss = self._policy.G(
                self._policy['vf_loss'], policy_update_feed_dict)
            ev = explained_variance(mb_rewards, mb_values)

            if self.t > 0 \
                    and self.t % self.checkpoint_freq == 0 \
                    and len(self._env.get_episode_rewards()) > 0:
                # log things
                episode_rewards = self._env.get_episode_rewards(
                    int(self.reward_len / self.num_envs))
                self.logger.set_metrics_for_iter(episode_rewards)
                self.logger.add_metric('timesteps_so_far', self.t)
                eta = (self.max_timesteps - self.t) /\
                    (self.t / round(time.time() - self.logger._start_time, 6))
                self.logger.add_metric(
                    'ETA',
                    str(rd(seconds=eta)))
                self.logger.add_metric('policy_loss', policy_loss)
                self.logger.add_metric('vf_loss', vf_loss)
                self.logger.add_metric('policy_entropy', policy_entropy)
                self.logger.add_metric('vf_explained_var', ev)
                self.logger.add_metric('env_id', self._env_id)
                self.logger.add_metric(
                    'last_saved_reward', self.last_saved_reward)
                self.logger.add_metric(
                    'policy_learning_rate', self.lr_schedule.value(self.t))
                self.logger.log()

            # save model if necessary
            if self.t % self.save_freq == 0 and self.t \
                    and self.logger._log_dir is not None \
                    and len(self._env.get_episode_rewards()) > 0:

                running_reward = np.mean(
                    self.logger._running_reward)
                if running_reward > self.last_saved_reward:
                    self.logger.logger.info(
                        'Saving best model, {} -> {}'.format(
                            self.last_saved_reward, running_reward))
                    self.save(self.logger._log_dir, 'best_agent')
                    self.last_saved_reward = running_reward

                self.logger.logger.info('Saving model for checkpoint')
                self.save(self.logger._log_dir)
