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
        'MountainCarContinuous-v0',
        normalize_obs=True)

    seed = 42
    env.seed(seed)
    tf_utils.set_global_seeds(seed)
    
    # env = NormalizedGymEnv('Acrobot-v1')
    # env = NormalizedGymEnv('Pendulum-v0')
    agent = TRPOAgent(
        env, discount_factor=0.99,
        policy_network=mlp, seed=0)
    agent.train(500, 0, n_steps=2048)


if __name__ == '__main__':
    main()
