"""
Sample run of ddqn
"""

from yarlp.agent.ddqn_agent import DDQNAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp, cnn
from functools import partial


def main():
    env = NormalizedGymEnv(
        'BreakoutNoFrameskip-v4',
        is_atari=True
    )

    agent = DDQNAgent(
        env=env,
        discount_factor=0.99,
        learning_start_timestep=10000,
        target_network_update_freq=10000,
        train_freq=1,
        exploration_final_eps=0.1,
        exploration_fraction=0.1,
        buffer_size=1000000,
        policy_learning_rate=1e-4,
        policy_network=cnn,
        max_timesteps=int(10e6),
        seed=123)
    agent.train()

    # env = NormalizedGymEnv(
    #     'CartPole-v0'
    # )
    # net = partial(mlp, hidden_units=[64])
    # agent = DDQNAgent(
    #     env=env,
    #     policy_network=net,
    #     policy_network_params={},
    #     policy_learning_rate=1e-3,
    #     max_timesteps=100000,
    #     buffer_size=50000,
    #     exploration_fraction=0.3,
    #     exploration_final_eps=0.05,
    #     prioritized_replay=True,
    #     double_q=True,
    #     train_freq=1,
    #     target_network_update_freq=500,
    #     learning_start_timestep=1000,
    #     seed=65)
    # agent.train()


if __name__ == '__main__':
    main()

