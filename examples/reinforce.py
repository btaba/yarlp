"""
Sample run with reinforce agent
"""

from yarlp.agent.pg_agents import REINFORCEAgent
from yarlp.agent.baseline_agents import RandomAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    env = NormalizedGymEnv(
        # 'MountainCarContinuous-v0'
        # 'CartPole-v1'
        # 'Walker2d-v1'
        'Acrobot-v1'
        # 'Pendulum-v0'
        # 'HalfCheetah-v1'
    )

    print(env.action_space)

    agent = REINFORCEAgent(
        env=env, discount_factor=0.99,
        policy_network=None,
        policy_learning_rate=0.01,
        entropy_weight=0,
        baseline_train_iters=5,
        baseline_model_learning_rate=1e-2,
        baseline_network=mlp,
        # baseline_network=None,
        seed=5)
    agent.train(500, n_steps=1024)


if __name__ == '__main__':
    main()
