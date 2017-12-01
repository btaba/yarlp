"""
Sample run with trpo agent
"""

from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp
from yarlp.utils import tf_utils


def main():
    # env = NormalizedGymEnv('CartPole-v1')
    env = NormalizedGymEnv(
        'Walker2d-v1',
        normalize_obs=False)

    seed = 123
    env.seed(seed)
    tf_utils.set_global_seeds(seed)

    # env = NormalizedGymEnv('Acrobot-v1')
    # env = NormalizedGymEnv('Pendulum-v0')
    agent = TRPOAgent(
        env, discount_factor=0.99,
        policy_network=mlp, seed=seed)
    agent.train(max_timesteps=10000000)


if __name__ == '__main__':
    main()
