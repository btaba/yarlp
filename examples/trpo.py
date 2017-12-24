"""
Sample run with trpo agent
"""

from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.model.networks import normc_initializer
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp
from yarlp.utils import tf_utils


def main():
    env = NormalizedGymEnv(
        'MountainCarContinuous-v0',
        # 'Walker2d-v1',
        # 'CartPole-v1',
        # 'Acrobot-v1',
        # 'Pendulum-v0',
        normalize_obs=True
        )

    print(env.action_space)

    agent = TRPOAgent(
        env, discount_factor=0.99,
        policy_network=mlp, seed=123,
        gae_lambda=0.98, cg_iters=10,
        cg_damping=1e-1, max_kl=1e-2,
        init_std=1.0, min_std=1e-6,
        baseline_train_iters=5,
        baseline_model_learning_rate=1e-2,
        baseline_network=mlp
        )
    agent.train(max_timesteps=1000000)


if __name__ == '__main__':
    main()
