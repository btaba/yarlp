"""
Sample run with trpo agent
"""

from yarlp.agent.trpo_agent import TRPOAgent
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.model.networks import mlp, cnn


def main():
    env = NormalizedGymEnv(
        'MountainCarContinuous-v0',
        # 'Walker2d-v1',
        # 'Swimmer-v1',
        # 'CartPole-v1',
        # 'Acrobot-v1',
        # 'Pendulum-v0',
        # 'PongNoFrameskip-v4',
        # normalize_obs=True,
        # is_atari=True
    )

    print(env.action_space)

    agent = TRPOAgent(
        env,
        policy_network=mlp, seed=123,
        baseline_train_iters=5,
        baseline_model_learning_rate=1e-3,
        baseline_network=mlp
    )
    agent.train(max_timesteps=1000000)


if __name__ == '__main__':
    main()
