"""
Sample run with trpo agent
"""

from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    # env = NormalizedGymEnv('CartPole-v1')
    env = NormalizedGymEnv(
        'MountainCarContinuous-v0',
        normalize_obs=True)
    # env = NormalizedGymEnv('Acrobot-v1')
    # env = NormalizedGymEnv('Pendulum-v0')
    agent = TRPOAgent(
        env, discount_factor=0.99,
        policy_network=mlp)
    agent.train(500, 0, n_steps=2048)


if __name__ == '__main__':
    main()
