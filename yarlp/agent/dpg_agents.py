"""
Deterministic Policy Gradients

[1] Silver, D. et. al. (2014). Deterministic Policy Gradient Algorithms.
    Proceedings of the 31st International Conference on Machine Learning
    (ICML-14), 387–395.

[2] Bengio, Y. (2016). Continuous control with deep reinforcement learning.
    Foundations and Trends® in Machine Learning, 2(1), 1–127.
    doi:10.1561/2200000006
"""

from yarlp.agent.base_agent import Agent
import tensorflow as tf

from yarlp.model.model_factories import ddpg_actor_critic_model_factory
from yarlp.utils.replay_buffer import ReplayBuffer


class DDPG(Agent):
    """Deep Deterministic Policy Gradient
    """
    def __init__(self, env, discount_factor=0.99,
                 actor_network=tf.contrib.layers.fully_connected,
                 actor_learning_rate=0.01,
                 critic_network=tf.contrib.layers.fully_connected,
                 critic_learning_rate=0.1,
                 minibatch_size=100,
                 replay_buffer_size=int(1e6),
                 tau=.001,
                 *args, **kwargs):
        super().__init__(env, discount_factor=discount_factor, *args, **kwargs)

        # Generate actor network (state), and critic network (state, action)
        self._policy = ddpg_actor_critic_model_factory(
            env, actor_network, critic_network, actor_learning_rate,
            critic_learning_rate)

        # Initialize target networks with same architecture and weights
        self._target = ddpg_actor_critic_model_factory(
            env, actor_network, critic_network, actor_learning_rate,
            critic_learning_rate)
        self._target.set_weights(self._policy.get_weights())

        # Initialize ReplayBuffer
        self.replay_buffer = ReplayBuffer(size=replay_buffer_size)
        self.minibatch_size = minibatch_size

        assert tau < 1 and tau > 0
        self.TAU = tau

    def train(self, num_training_steps):
        # loop through episodes
        for i in range(num_training_steps):

            # Initialize random process for action exploration

            obs = self.get_state(self._env.reset())
            for t in range(self.num_max_rollout_steps):
                # get action, take action, store transition in ReplayBuffer
                a = self._policy.predict(
                    obs, output_name='output:action', input_name='input:state')
                (obs_prime, reward, done, _) = self._env.step(a)
                obs_prime = self.get_state(obs_prime)
                self.replay_buffer.append(obs, a, reward, obs_prime, done)

                if done:
                    break

                # Sample random minibatch from ReplayBuffer
                if self.replay_buffer.size >= self.minibatch_size:
                    minibatch = self.replay_buffer.get_random_minibatch(
                        minibatch_size=self.minibatch_size)

                    target_actions = self._target.predict(
                        minibatch.next_state, output_name='output:action',
                        input_name='input:state')

                    # get target Q
                    feed = {self._target.state: minibatch.next_state,
                            self._target.action_input: target_actions}
                    target_q = self._target.run_op(self._target.q_value, feed)

                    term = 1 - minibatch.terminal
                    y = minibatch.reward + \
                        self._discount_factor * target_q * term

                    # update critic
                    feed = {self._policy.state: minibatch.state,
                            self._policy.action_input: minibatch.action,
                            self._policy.td_value: y}
                    _, loss = self._policy.run_op(
                        [self._policy.critic_optimizer,
                            self._policy.critic_loss], feed)

                    # update actor
                    feed = {self._policy.state: minibatch.state}
                    _, loss = self._policy.run_op(
                        [self._policy.actor_optimizer,
                            self._policy.actor_loss], feed)

                    # update target network
                    policy_weights = self._policy.get_weights()
                    target_weights = self._target.get_weights()
                    new_target_weights = [
                        p * self.TAU + t * (1 - self.TAU)
                        for p, t in zip(policy_weights, target_weights)]
                    self._target.set_weights(new_target_weights)
