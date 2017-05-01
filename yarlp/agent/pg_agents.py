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

from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import value_function_model_factory
from yarlp.model.model_factories import discrete_pg_model_factory
# from yarlp.model.model_factories import continuous_gaussian_pg_model_factory
# from yarlp.utils.logger import logger

import numpy as np
import tensorflow as tf
# import warnings


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient for discrete action spaces

    Parameters
    ----------
    policy_model : model.Model

    baseline_network : if None, we us no baseline

    """
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_learning_rate=0.01,
                 baseline_network=tf.contrib.layers.fully_connected,
                 value_learning_rate=0.01, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self._policy = discrete_pg_model_factory(
            env, policy_network, policy_learning_rate)

        if not baseline_network:
            # No baseline
            self._baseline_model = None
        else:
            # Baseline can be any function,
            # as long as it does not vary with actions
            self._baseline_model = value_function_model_factory(
                env, network=baseline_network,
                learning_rate=value_learning_rate)

    def train(self, num_train_steps, num_test_steps=0):
        """

        Parameters
        ----------
        num_training_steps : integer
            Total number of training steps

        Returns
        ----------
        total_reward_per_training_episode : list
            total reward obtained after each training episode

        """
        for i in range(num_train_steps):
            # execute an episode
            rollout = self.rollout()

            actions = []
            states = []
            advantages = []

            for t, r in enumerate(rollout.rewards):
                # update the weights for policy model
                discounted_rt = self.get_discounted_cumulative_reward(
                    rollout.rewards[t:])

                baseline = 0
                if self._baseline_model:
                    self._baseline_model.update(
                        rollout.states[t], discounted_rt)
                    baseline = self._baseline_model.predict(
                        np.array(rollout.states[t])).flatten()

                advantage = discounted_rt - baseline
                states.append(rollout.states[t])
                actions.append(rollout.actions[t])
                advantages.append(advantage)

            # batch update the policy
            self._policy.update(
                states, advantages, actions)
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
