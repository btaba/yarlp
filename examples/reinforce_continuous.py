"""
Sample run with reinforce agent
"""

from yarlp.agent import REINFORCEAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    # env = NormalizedGymEnv("CartPole-v1")
    env = NormalizedGymEnv('MountainCarContinuous-v0')
    # env = NormalizedGymEnv('Acrobot-v1')
    # env = NormalizedGymEnv('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    agent = REINFORCEAgent(
        env, discount_factor=0.99,
        policy_network=mlp,
        policy_learning_rate=0.01,
        adaptive_std=False,
        entropy_weight=0.)
    agent.train(500, 0, n_steps=4000, whiten_advantages=True, render=False)


if __name__ == '__main__':
    main()
