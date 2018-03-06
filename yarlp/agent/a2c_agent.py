"""
A2C - synchronous advantage actor critic
"""
import numpy as np
from yarlp.agent.base_agent import Agent
from yarlp.model.networks import cnn
from yarlp.utils.experiment_utils import get_network
from yarlp.model.model_factories import value_function_model_factory
from yarlp.model.model_factories import pg_model_factory


class A2C(Agent):

    def __init__(
            self, env,
            policy_network=None,
            policy_network_params={},
            policy_learning_rate=0.01,
            value_fn_learning_rate=0.001,
            entropy_weight=0,
            model_file_path=None,
            adaptive_std=False,
            init_std=1.0,
            min_std=1e-6,
            n_steps=5,
            max_timesteps=1000000,
            # gae_lambda=1.,
            *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        
        if policy_network is None:
            policy_network = cnn

        policy_network = get_network(policy_network, policy_network_params)

        self._policy = pg_model_factory(
            env, network=policy_network,
            network_params=policy_network_params,
            learning_rate=policy_learning_rate,
            entropy_weight=entropy_weight,
            min_std=min_std, init_std=init_std, adaptive_std=adaptive_std,
            model_file_path=model_file_path)

        self._value_fn = value_function_model_factory(
            env, network=policy_network,
            network_params=policy_network_params,
            learning_rate=value_fn_learning_rate,
            model_file_path=model_file_path)

        self.tf_object_attributes.add('_policy', '_value_fn')
        policy_weight_sums = sum(
            [np.sum(a) for a in self._policy.get_weights()])
        self.logger.logger.info(
            'Policy network weight sums: {}'.format(policy_weight_sums))

        self.n_steps = n_steps
        self.max_timesteps = max_timesteps
        # self._gae_lambda = gae_lambda

    def train(self):
        
        self.t = 0

        obs = self.env.reset()

        while self.t < self.max_timesteps:

            # do n-steps in each env
            for _ in range(self.n_steps):
                actions = self.get_batch_actions(obs)
                obs, rewards, dones, _ = self.env.step(actions)
                
            # calculate advantage
            # fit the value fn
            # fit the policy network

            # log stuff











