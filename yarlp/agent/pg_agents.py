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
import numpy as np
import tensorflow as tf

from yarlp.agent.base_agent import Agent
from yarlp.model.model_factories import pg_model_factory
from yarlp.model.model_factories import value_function_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline
from yarlp.utils.experiment_utils import get_network


class REINFORCEAgent(Agent):
    """
    REINFORCE - Monte Carlo Policy Gradient for discrete action spaces

    Parameters
    ----------
    env : gym.env

    policy_network : model.Model

    policy_learning_rate : float, the learning rate for the policy_network

    baseline_network : if None, we us no baseline
        otherwise we use a LinearFeatureBaseline as default
        you can also pass in a function as a tensorflow network which
        gets built by the value_function_model_factory

    entropy_weight : float, coefficient on entropy in the policy gradient loss

    model_file_path : str, file path for the policy_network
    """
    def __init__(self, env,
                 policy_network=tf.contrib.layers.fully_connected,
                 policy_network_params={},
                 policy_learning_rate=0.01,
                 baseline_network=LinearFeatureBaseline(),
                 baseline_model_learning_rate=0.01,
                 entropy_weight=0,
                 model_file_path=None,
                 adaptive_std=False,
                 init_std=1.0,
                 min_std=1e-6,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        policy_path = None
        if model_file_path:
            policy_path = os.path.join(model_file_path, 'pg_policy')

        policy_network = get_network(policy_network, policy_network_params)

        self._policy = pg_model_factory(
            env, network=policy_network, network_params=policy_network_params,
            learning_rate=policy_learning_rate, entropy_weight=entropy_weight,
            min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
            model_file_path=policy_path)

        if isinstance(baseline_network, LinearFeatureBaseline)\
                or baseline_network is None:
            self._baseline_model = baseline_network
        else:
            self._baseline_model = value_function_model_factory(
                env, policy_network,
                learning_rate=baseline_model_learning_rate)

    def save_models(self, path):
        ppath = os.path.join(path, 'pg_policy')
        self._policy.save(ppath)

    def train(self, num_train_steps=10, num_test_steps=0,
              n_steps=1000, render=False, whiten_advantages=True):
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

        whiten_advantages : bool, whether to whiten the advantages

        render : bool, whether to render episodes in a video

        Returns
        ----------
        None
        """
        for i in range(num_train_steps):
            # execute an episode
            rollouts = self.rollout_n_steps(n_steps, render=render)

            actions = []
            states = []
            advantages = []
            discounted_rewards = []

            for rollout in rollouts:
                discounted_reward = self.get_discounted_reward_list(
                    rollout.rewards)

                baseline_pred = np.zeros_like(discounted_reward)
                if self._baseline_model:
                    baseline_pred = self._baseline_model.predict(
                        np.array(rollout.states)).flatten()

                baseline_pred = np.append(baseline_pred, 0)
                advantage = rollout.rewards + self._discount *\
                    baseline_pred[1:] - baseline_pred[:-1]
                advantage = self.get_discounted_reward_list(
                    advantage)
                # advantage = discounted_reward - baseline_pred

                advantages = np.concatenate([advantages, advantage])
                states.append(rollout.states)
                actions.append(rollout.actions)
                discounted_rewards = np.concatenate(
                    [discounted_rewards, discounted_reward])

            states = np.concatenate([s for s in states]).squeeze()
            actions = np.concatenate([a for a in actions])

            if whiten_advantages:
                advantages = (advantages - np.mean(advantages)) /\
                    (np.std(advantages) + 1e-8)

            # batch update the baseline model
            if isinstance(self._baseline_model, LinearFeatureBaseline):
                self._baseline_model.fit(states, discounted_rewards)
            elif hasattr(self._baseline_model, 'G'):
                self._baseline_model.update(
                    states, discounted_rewards)

            # update the policy
            loss = self._policy.update(
                states, advantages.squeeze(), actions.squeeze())
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
