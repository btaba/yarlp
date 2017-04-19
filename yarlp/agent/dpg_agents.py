# """
# Deterministic Policy Gradients

# [1] Silver, D. et. al. (2014). Deterministic Policy Gradient Algorithms.
#     Proceedings of the 31st International Conference on Machine Learning
#     (ICML-14), 387–395.

# [2] Bengio, Y. (2016). Continuous control with deep reinforcement learning.
#     Foundations and Trends® in Machine Learning, 2(1), 1–127.
#     doi:10.1561/2200000006
# """
# import numpy as np
# import tensorflow as tf
# from yarlp.agent.base_agent import Agent

# from yarlp.model.model_factories import ddpg_actor_critic_model_factory
# from yarlp.utils.replay_buffer import ReplayBuffer
# from yarlp.utils.exploration_noise import OrnsteinUhlenbeck


# class DDPG(Agent):
#     """Deep Deterministic Policy Gradient
#     """
#     def __init__(self, env, discount_factor=0.99,
#                  actor_network=tf.contrib.layers.fully_connected,
#                  actor_learning_rate=1e-3,
#                  critic_network=tf.contrib.layers.fully_connected,
#                  critic_learning_rate=1e-1,
#                  minibatch_size=32,
#                  min_buffer_size=2000,
#                  replay_buffer_size=int(1e6),
#                  tau=.001,
#                  exploration_noise=OrnsteinUhlenbeck(0, .15, .3,
#                                                      n_actions=1, dt=1),
#                  *args, **kwargs):
#         super().__init__(env, discount_factor=discount_factor, *args, **kwargs)

#         # Generate actor network (state), and critic network (state, action)
#         self._policy = ddpg_actor_critic_model_factory(
#             env, actor_network, critic_network, actor_learning_rate,
#             critic_learning_rate)

#         # Initialize target networks with same architecture and weights
#         self._target = ddpg_actor_critic_model_factory(
#             env, actor_network, critic_network, actor_learning_rate,
#             critic_learning_rate)
#         self._target.set_weights(self._policy.get_weights())

#         # Initialize ReplayBuffer
#         self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
#         self.minibatch_size = minibatch_size
#         assert min_buffer_size >= minibatch_size
#         self.min_buffer_size = min_buffer_size

#         assert tau < 1 and tau > 0
#         self.TAU = tau
#         self.exploration_noise = exploration_noise

#     def train(self, num_train_steps):
#         # loop through episodes
#         for i in range(num_train_steps):

#             self.exploration_noise.reset()

#             obs = self.get_state(self._env.reset())

#             for t in range(self._env.spec.timestep_limit):
#                 # get action, take action, store transition in ReplayBuffer
#                 a = self._policy.predict(
#                     obs, output_name='output:', input_name='input:')
#                 a_new = self.exploration_noise.add_noise_to_action(
#                     self._env, a)

#                 (obs_prime, reward, done, _) = self._env.step(a_new)
#                 obs_prime = self.get_state(obs_prime)
#                 self.replay_buffer.append(obs, a_new, reward, obs_prime, done)
#                 obs = obs_prime

#                 if t and t % 10 == 0:
#                     print(a, a_new)
#                     self._env.render()

#                 if done:
#                     print(t)
#                     break

#                 # Sample random minibatch from ReplayBuffer
#                 if self.replay_buffer.size >= self.min_buffer_size:
#                     minibatch = self.replay_buffer.get_random_minibatch(
#                         batch_size=self.minibatch_size)

#                     target_actions = self._target.predict(
#                         minibatch.next_state, output_name='output:',
#                         input_name='input:')

#                     # get target Q
#                     feed = {self._target.state: minibatch.next_state,
#                             self._target.action_input:
#                                 np.expand_dims(target_actions, 1)}
#                     target_q = self._target.run_op(self._target.q_value, feed)
#                     target_q = target_q.flatten()

#                     term = 1 - minibatch.terminal
#                     y = minibatch.reward + \
#                         self._discount * np.multiply(target_q, term)

#                     # update critic
#                     feed = {self._policy.state: minibatch.state,
#                             self._policy.action_input: minibatch.action,
#                             self._policy.td_value: y}
#                     _, loss = self._policy.run_op(
#                         [self._policy.critic_optimizer_op,
#                             self._policy.critic_loss], feed)

#                     # update actor
#                     feed = {self._policy.state: minibatch.state}
#                     _, loss = self._policy.run_op(
#                         [self._policy.actor_optimizer_op,
#                             self._policy.actor_loss], feed)

#                     # update target network
#                     policy_weights = self._policy.get_weights()
#                     target_weights = self._target.get_weights()
#                     new_target_weights = [
#                         p * self.TAU + t * (1 - self.TAU)
#                         for p, t in zip(policy_weights, target_weights)]
#                     self._target.set_weights(new_target_weights)
