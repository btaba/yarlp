"""
Sample run with reinforce agent
"""

from yarlp.agent.cem_agent import CEMAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp


def main():
    env = NormalizedGymEnv(
        'MountainCarContinuous-v0'
        # 'CartPole-v1'
        # 'Acrobot-v1'
        # 'Pendulum-v0'
        # 'HalfCheetah-v1'
    )
    agent = CEMAgent(
        env, discount_factor=0.99,
        n_weight_samples=100,
        init_var=1., best_pct=0.2,
        policy_network=None,
        policy_network_params={},
        model_file_path=None,
        min_std=1e-6, init_std=1.0, adaptive_std=False,
        seed=5)
    agent.train(100)


if __name__ == '__main__':
    main()
