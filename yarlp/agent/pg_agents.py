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
import tensorflow as tf

from yarlp.agent.base_agent import BatchAgent
from yarlp.model.model_factories import pg_model_factory
from yarlp.model.model_factories import value_function_model_factory
from yarlp.model.linear_baseline import LinearFeatureBaseline
from yarlp.utils.experiment_utils import get_network


class REINFORCEAgent(BatchAgent):
    """
    REINFORCE - Monte Carlo Policy Gradient for discrete action spaces

    Parameters
    ----------
    env : gym.env

    policy_network : model.Model

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
                 baseline_network=LinearFeatureBaseline(),
                 baseline_model_learning_rate=0.01,
                 entropy_weight=0,
                 model_file_path=None,
                 adaptive_std=False,
                 init_std=1.0,
                 min_std=1e-6, gae_lambda=1.,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        policy_path = None
        if model_file_path:
            policy_path = os.path.join(model_file_path, 'pg_policy')

        policy_network = get_network(policy_network, policy_network_params)

        self._policy = pg_model_factory(
            env, network=policy_network, network_params=policy_network_params,
            entropy_weight=entropy_weight,
            min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
            model_file_path=policy_path)
        self._gae_lambda = gae_lambda

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

    def update(self, path):
        loss = self._policy.update(
            path['states'], path['advantages'],
            path['actions'])
        self.logger.add_metric('policy_loss', loss)
