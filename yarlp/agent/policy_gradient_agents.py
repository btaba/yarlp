"""
REINFORCE Agent and Policy Gradient (PG) Actor Critic Agent
"""

from functools import partial
from yarlp.agent.base_agent import Agent
from yarlp.model.tf_model import Model
from yarlp.model.model_factories import value_function_model_factory

import numpy as np
import tensorflow as tf


class PGAgent(Agent):
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_learning_rate=0.01,
                 action_space='discrete',
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self._policy = PGAgent.pg_model_factory(
            env, policy_network, policy_learning_rate,
            action_space)

    @staticmethod
    def pg_model_factory(
            env, network, learning_rate,
            action_space):
        """ Vanilla policy gradient for discrete and continuous action spaces

        type : str
            whether the action space is 'continuous' or 'discrete'
        """

        def build_graph(model, network, action_space, lr):
            input_node = model.add_input()

            model.state = input_node
            model.Return = tf.placeholder(
                dtype=tf.float32, shape=(None,), name='return')
            model.learning_rate = lr

            # Policy gradient stuff
            if action_space == 'discrete':
                # Softmax policy for discrete action spaces
                network = partial(network, activation_fn=tf.nn.softmax)
                output_node = model.add_output(network)

                model.action = tf.placeholder(
                    dtype=tf.int32, shape=(None,), name='action')
                action_probability = tf.gather(
                    tf.squeeze(output_node), model.action)

                model.loss = -tf.log(action_probability) * model.Return

                model.optimizer = tf.train.AdamOptimizer(
                    learning_rate=lr)

                model.log_pi = tf.log(action_probability)
                model.create_gradient_ops_for_node(model.log_pi)
            elif action_space == 'continuous':
                # Gaussian policy is natural to use in continuous action spaces
                # http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl7.pdf
                network = partial(network, activation_fn=None)
                model.mu = model.add_output(network, name='mean')

                # std dev must always be positive
                model.sigma = model.add_output(network, name='std_dev')
                model.sigma = tf.exp(model.sigma) + 1e-6
                # model.sigma = tf.log(tf.exp(model.sigma) + 1) + 1e-6

                model.normal_dist = tf.contrib.distributions.Normal(
                    model.mu, model.sigma)
                model.action = tf.squeeze(model.normal_dist.sample([1]))
                model.action = tf.clip_by_value(
                    model.action, model._env.action_space.low[0],
                    model._env.action_space.high[0])
                model.add_output_node(model.action)

                model.loss = -model.normal_dist.log_prob(
                    model.action) * model.Return
                model.loss -= 0.1 * model.normal_dist.entropy()

                model.optimizer = tf.train.AdamOptimizer(
                    learning_rate=lr)
                model.log_pi = model.normal_dist.log_prob(model.action)
                model.create_gradient_ops_for_node(model.log_pi)
            else:
                raise ValueError('%s is not a valid action_space'
                                 % action_space)

        def build_update_feed_dict(model, state, return_, action):
            feed_dict = {model.state: np.expand_dims(np.array(state), 0),
                         model.Return: [return_], model.action: [action]}
            return feed_dict

        build_graph = partial(build_graph, network=network,
                              action_space=action_space,
                              lr=learning_rate)

        return Model(env, build_graph, build_update_feed_dict)


class REINFORCEAgent(PGAgent):
    """
    REINFORCE - Monte Carlo Policy Gradient
    and Policy Gradient with Function Approximation
    [1] Simple statistical gradient-following algorithms for connectionist
        reinforcement learning (Williams, 1992)
        pdf: http://link.springer.com/article/10.1007/BF00992696

    [2] Sutton, R. S., Mcallester, D., Singh, S., & Mansour, Y. (1999).
        Policy Gradient Methods for Reinforcement Learning with
        Function Approximation. Advances in Neural Information Processing
        Systems 12, 1057–1063. doi:10.1.1.37.9714

    Parameters
    ----------
    policy_model : tf_model.Model

    value_model : either 'linear' or 'average', defaults to 'average'

    """
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_learning_rate=0.01, action_space='discrete',
                 value_model='average',
                 value_learning_rate=0.01, *args, **kwargs):
        super().__init__(env, policy_network, policy_learning_rate,
                         action_space, *args, **kwargs)

        if value_model == 'linear':
            # Policy Gradient with function approximation
            # theoretically the state-value approximation should
            # be linear in policy features if the policy model is a softmax
            # due to Theorem 2 in [2]
            self._value_model = value_function_model_factory(
                self._policy.env, network=tf.contrib.layers.fully_connected,
                learning_rate=value_learning_rate)
        else:
            # Classic REINFORCE from [1]
            self._value_model = None

    def train(self, num_training_steps, with_baseline=True):
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
        total_reward_per_training_episode = []
        for i in range(num_training_steps):
            # execute an episode
            rollout = self.rollout()

            # save average reward for this training step for reporting
            total_reward_per_training_episode.append(np.sum(rollout.rewards))

            for t, r in enumerate(rollout.rewards):
                # update the weights for policy model
                discounted_rt = self.get_discounted_cumulative_reward(
                    rollout.rewards[t:])

                baseline = 0
                if with_baseline:
                    if self._value_model:
                        self._value_model.update(
                            rollout.states[t], discounted_rt)
                        baseline = self._value_model.predict(
                            np.array(rollout.states[t]))[0]
                    else:
                        baseline = np.mean(rollout.rewards[t:])

                advantage = discounted_rt - baseline
                self._policy.update(
                    rollout.states[t], advantage, rollout.actions[t])

        return total_reward_per_training_episode


class ActorCriticPG(PGAgent):
    """Multi-step actor critic with policy gradients.
    Boostrapping returns introduces bias and can be difficult to tune.

    Parameters
    ----------
    """

    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_learning_rate=0.01, action_space='discrete',
                 value_model_learning_rate=0.1,
                 lambda_p=1, lambda_v=1, *args, **kwargs):
        super().__init__(env, policy_network, policy_learning_rate,
                         action_space, *args, **kwargs)

        self._value_model = value_function_model_factory(
            self._policy.env, network=tf.contrib.layers.fully_connected,
            learning_rate=value_model_learning_rate)

        self._lambda_p = lambda_p
        self._lambda_v = lambda_v

    def train(self, num_training_steps):
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
        total_reward_per_training_episode = []
        for i in range(num_training_steps):

            # Make eligibility traces for each weight
            e_v = self._value_model.get_weights()
            e_v = [np.zeros_like(e) for e in e_v]
            e_p = self._policy.get_weights()
            e_p = [np.zeros_like(e) for e in e_p]

            # execute an episode
            discount = 0
            total_rewards = 0
            obs = self._env.reset()
            for t in range(self.num_max_rollout_steps):
                action = self.get_action(obs)

                (obs_prime, reward, done, _) = self._env.step(action)
                # self._env.render()
                total_rewards += reward

                v_prime = 0 if done else self._value_model.predict(
                    np.array(obs_prime))[0]
                v = self._value_model.predict(np.array(obs))[0]

                td_target = reward + self._discount * v_prime
                td_error = td_target - v

                feed = {self._value_model.state:
                        np.expand_dims(np.array(obs), 0)}
                grads_v = self._value_model.get_gradients(
                    self._value_model.value.name, feed)
                e_v = [e * self._lambda_v + discount * g[0]
                       for e, g in zip(e_v, grads_v)]

                w_v = self._value_model.get_weights()
                w_v = [w + self._value_model.learning_rate * td_error * e
                       for w, e in zip(w_v, e_v)]
                self._value_model.set_weights(w_v)

                feed = {self._policy.state: np.expand_dims(np.array(obs), 0),
                        self._policy.action.name: [action]}
                grads_p = self._policy.get_gradients(
                    self._policy.log_pi.name, feed)
                e_p = [e * self._lambda_p + discount * g[0]
                       for e, g in zip(e_p, grads_p)]

                w_p = self._policy.get_weights()
                w_p = [w + self._policy.learning_rate * td_error * e
                       for w, e in zip(w_p, e_p)]
                self._policy.set_weights(w_p)

                discount *= self._discount
                obs = obs_prime

                if done:
                    break
            print(t, total_rewards)

            total_reward_per_training_episode.append(total_rewards)

        return total_reward_per_training_episode
