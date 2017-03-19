"""
Grid search on ActorCriticPG for MountainCar
"""

import gym
import numpy as np

from itertools import product
from yarlp.model.model_factories import policy_gradient_model_factory
from yarlp.agent.policy_gradient_agents import ActorCriticPG


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


grid_dict = {
    'policy_learning_rate': [0.01, 0.5, 1],
    'lambda_p': [0.5, 1],
    'lambda_v': [1],
    'value_learning_rate': [0.1, 1]
}

grid_dict = {
    'policy_learning_rate': [0.5],
    'lambda_p': [1],
    'lambda_v': [1],
    'value_learning_rate': [1]
}
grid_vals = dict_product(grid_dict)


env = gym.make('MountainCarContinuous-v0')
for idx, g in enumerate(grid_vals):
    rewards = []
    for _ in range(3):
        pm = policy_gradient_model_factory(
            env, action_space='continuous',
            learning_rate=g['policy_learning_rate'])
        agent = ActorCriticPG(
            pm, num_max_rollout_steps=10000,
            discount_factor=1, lambda_p=g['lambda_p'],
            lambda_v=g['lambda_v'],
            value_model_learning_rate=g['value_learning_rate'])
        r = agent.train(num_training_steps=1)
        rewards.append(np.mean(r))

    grid_vals[idx]['reward'] = np.mean(rewards)

print(sorted(grid_vals, key=lambda x: x['reward']))
