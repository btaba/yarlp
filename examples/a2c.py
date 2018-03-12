"""
Sample run of A2C
"""

from yarlp.agent.a2c_agent import A2CAgent
from yarlp.utils.env_utils import ParallelEnvs
from yarlp.model.networks import mlp, cnn
from functools import partial


def main():
    env = ParallelEnvs('BeamRiderNoFrameskip-v4', 4)
    # env = ParallelEnvs('CartPole-v1', 4, is_atari=False)

    # net = partial(mlp, hidden_units=[64])
    agent = A2CAgent(
        env=env,
        policy_network=cnn,
        # policy_network=net,
        policy_network_params={},
        entropy_weight=0.001,
        max_timesteps=1000000,
        checkpoint_freq=1000,
        policy_learning_rate_schedule=[[0, 1e-4], [1e6, 1e-5]],
        seed=65)
    agent.train()


if __name__ == '__main__':
    main()
