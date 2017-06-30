"""
REINFORCE Agent and Policy Gradient (PG) Actor Critic Agent

[1] Simple statistical gradient-following algorithms for connectionist
    reinforcement learning (Williams, 1992)
    pdf: http://link.springer.com/article/10.1007/BF00992696

[2] Sutton, R. S., Mcallester, D., Singh, S., & Mansour, Y. (1999).
    Policy Gradient Methods for Reinforcement Learning with
    Function Approximation. Advances in Neural Information Processing
    Systems 12, 1057–1063. doi:10.1.1.37.9714

[3] Degris, T., Pilarski, P., & Sutton, R. (2012). Model-free reinforcement
    learning with continuous action in practice. … Control Conference (ACC),
    2177–2182. doi:10.1109/ACC.2012.6315022
"""

import os
from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import discrete_pg_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline

import numpy as np
import tensorflow as tf
from yarlp.utils.experiment_utils import get_network


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient for discrete action spaces

    Parameters
    ----------
    env : gym.env
    policy_network : model.Model
    policy_learning_rate : float, the learning rate for the policy_network
    use_baseline : if None, we us no baseline
        otherwise we use a LinearFeatureBaseline
    entropy_weight : float, coefficient on entropy in the policy gradient loss
    model_file_path : str, file path for the policy_network
    """
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 policy_learning_rate=0.01,
                 use_baseline=True,
                 entropy_weight=0,
                 model_file_path=None,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        policy_path = None
        if model_file_path:
            policy_path = os.path.join(model_file_path, 'pg_policy')

        policy_network = get_network(policy_network, policy_network_params)

        self._policy = discrete_pg_model_factory(
            env, policy_network, policy_learning_rate, entropy_weight,
            model_file_path=policy_path)

        if use_baseline:
            self._baseline_model = LinearFeatureBaseline()
        else:
            self._baseline_model = None

    def save_models(self, path):
        ppath = os.path.join(path, 'pg_policy')
        self._policy.save(ppath)

    def train(self, num_train_steps=10, num_test_steps=0, n_steps=1000):
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

        Returns
        ----------
        None
        """
        for i in range(num_train_steps):
            # execute an episode
            rollouts = self.rollout_n_steps(n_steps)

            actions = []
            states = []
            advantages = []
            discounted_rewards = []
            baseline_preds = []

            for rollout in rollouts:
                discounted_reward = self.get_discounted_reward_list(
                    rollout.rewards)

                baseline_pred = np.zeros_like(discounted_reward)
                if self._baseline_model:
                    baseline_pred = self._baseline_model.predict(
                        np.array(rollout.states)).flatten()

                # calculate and whiten the advantages
                advantage = discounted_reward - baseline_pred
                advantage = (advantage - np.mean(advantage)) /\
                    (np.std(advantage) + 1e-8)

                baseline_preds = np.concatenate(
                    [baseline_preds, baseline_pred])
                advantages = np.concatenate([advantages, advantage])
                states.append(rollout.states)
                actions.append(rollout.actions)
                discounted_rewards = np.concatenate(
                    [discounted_rewards, discounted_reward])

            states = np.concatenate([s for s in states]).squeeze()
            actions = np.concatenate([a for a in actions])

            # batch update the baseline and the policy
            if self._baseline_model:
                self._baseline_model.fit(states, discounted_rewards)

            loss = self._policy.update(
                states, advantages.squeeze(), actions)
            self.logger.add_metric('policy_loss', loss)
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

        return


# class ActorCriticPG(Agent):
#     """Multi-step actor critic with policy gradients from [3] for
#     continuous and discrete action spaces.
#     Boostrapping returns introduces bias and can be difficult to tune.

#     Parameters
#     ----------
#     """

#     def __init__(self, env, discount_factor=0.99,
#                  policy_network=tf.contrib.layers.fully_connected,
#                  policy_learning_rate=0.01, action_space='continuous',
#                  baseline_network=tf.contrib.layers.fully_connected,
#                  value_model_learning_rate=0.1,
#                  lambda_p=0.5, lambda_v=0.9,
#                  *args, **kwargs):
#         super().__init__(env, discount_factor=discount_factor, *args, **kwargs)

#         input_shape = (None, self.get_state(
#             env.observation_space.sample()).shape[0])

#         if action_space == 'discrete':
#             self._policy = discrete_pg_model_factory(
#                 env, policy_network,
#                 policy_learning_rate, input_shape)
#         elif action_space == 'continuous':
#             self._policy = continuous_gaussian_pg_model_factory(
#                 env, policy_learning_rate, input_shape)
#             warnings.warn('ActorCriticPG may result in numerical instability '
#                           'with Gaussian policy', UserWarning)
#         else:
#             raise ValueError('%s is an invalid action_space' % action_space)

#         self._value_model = value_function_model_factory(
#             env, network=baseline_network,
#             learning_rate=value_model_learning_rate,
#             input_shape=input_shape)

#         self._lambda_p = lambda_p
#         self._lambda_v = lambda_v
#         self._action_space = action_space

#     def train(self, num_training_steps):
#         """

#         Parameters
#         ----------
#         num_training_steps : integer
#             Total number of training steps

#         Returns
#         ----------
#         total_reward_per_training_episode : list
#             total reward obtained after each training episode

#         """
#         total_reward_per_training_episode = []
#         for i in range(num_training_steps):

#             # Make eligibility traces for each weight
#             e_v = self._value_model.get_weights()
#             e_v = [np.zeros_like(e) for e in e_v]
#             e_p = self._policy.get_weights()
#             e_p = [np.zeros_like(e) for e in e_p]

#             # execute an episode
#             total_rewards = 0
#             obs = self._env.reset()
#             obs = self.get_state(obs)
#             for t in range(self.num_max_rollout_steps):
#                 action = self.get_action(obs)

#                 (obs_prime, reward, done, _) = self._env.step(action)

#                 total_rewards += reward
#                 obs_prime = self.get_state(obs_prime)

#                 # Get the TD error
#                 v_prime = 0 if done else self._value_model.predict(
#                     obs_prime)[0]
#                 v = self._value_model.predict(obs)[0]
#                 td_target = reward + self._discount * v_prime
#                 td_error = td_target - v

#                 # Update the value function
#                 feed = {self._value_model.state:
#                         np.expand_dims(obs, 0)}
#                 grads_v = self._value_model.get_gradients(
#                     self._value_model.value.name, feed)

#                 e_v = [e * self._lambda_v * self._discount + g[0]
#                        for e, g in zip(e_v, grads_v)]
#                 w_v = self._value_model.get_weights()
#                 w_v = [w + self._value_model.learning_rate * td_error * e
#                        for w, e in zip(w_v, e_v)]
#                 self._value_model.set_weights(w_v)

#                 # Update the policy function
#                 feed = {self._policy.state: np.expand_dims(obs, 0),
#                         self._policy.action.name: [action]}
#                 grads_p = self._policy.get_gradients(
#                     self._policy.log_pi.name, feed)

#                 assert np.all([not np.any(np.isnan(g)) and
#                               not np.any(np.isinf(g)) for g in grads_p])

#                 e_p = [e * self._lambda_p * self._discount + g[0]
#                        for e, g in zip(e_p, grads_p)]
#                 w_p = self._policy.get_weights()
#                 w_p = [w + self._policy.learning_rate * td_error * e
#                        for w, e in zip(w_p, e_p)]
#                 self._policy.set_weights(w_p)

#                 obs = obs_prime
#                 if done:
#                     break

#             print(t, total_rewards)

#             total_reward_per_training_episode.append(total_rewards)

#         return total_reward_per_training_episode
