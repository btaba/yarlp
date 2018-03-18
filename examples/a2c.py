"""
Sample run of A2C
"""

from yarlp.agent.a2c_agent import A2CAgent
from yarlp.utils.env_utils import ParallelEnvs
from yarlp.model.networks import mlp, cnn
from functools import partial


def main():
    # env = ParallelEnvs('PongNoFrameskip-v4', 12)
    env = ParallelEnvs('CartPole-v1', 4, is_atari=False)

    net = partial(mlp, hidden_units=[64])
    agent = A2CAgent(
        env=env,
        # policy_network=cnn,
        policy_network=net,
        policy_network_params={},
        entropy_weight=0.01,
        max_timesteps=1000000,
        checkpoint_freq=10000,
        policy_learning_rate_schedule=[[0, 7e-4], [1e6, 1e-16]],
        seed=65)
    agent.train()


if __name__ == '__main__':
    main()
