"""
Sample run with reinforce agent
"""

from yarlp.agent.pg_agents import REINFORCEAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    # env = NormalizedGymEnv('CartPole-v2')
    env = NormalizedGymEnv('HalfCheetah-v1')
    env.spec.timestep_limit = 150
    # env = NormalizedGymEnv('MountainCarContinuous-v0')
    # env = NormalizedGymEnv('Acrobot-v1')
    # env = NormalizedGymEnv('Pendulum-v0')
    agent = REINFORCEAgent(
        env, discount_factor=0.9,
        policy_network=mlp,
        policy_learning_rate=0.01,
        entropy_weight=0,
        baseline_network=None,
        seed=1)
    agent.train(500, 0, n_steps=4000)


if __name__ == '__main__':
    main()
