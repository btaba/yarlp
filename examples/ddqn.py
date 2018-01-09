"""
Sample run of ddqn
"""

from yarlp.agent.ddqn_agent import DDQNAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    env = NormalizedGymEnv(
        # 'CartPole-v1'
        'PongNoFrameskip-v4',
        is_atari=True
    )

    print(env.action_space)

    agent = DDQNAgent(
        env=env,
        # policy_network=mlp,
        seed=345)
    agent.train()


if __name__ == '__main__':
    main()
