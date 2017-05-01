import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym.core import Env


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


# class FixedIntervalVideoSchedule(object):
#     def __init__(self, interval):
#         self.interval = interval

#     def __call__(self, count):
#         return count % self.interval == 0


class GymEnv(Env):
    """Taken from rllab gym_env.py
    """
    def __init__(self, env_name, video=False,
                 log_dir=None,
                 force_reset=False):
        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id
        self.spec = env.spec
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        assert isinstance(video, bool)
        if log_dir is None:
            self.monitoring = False
        else:
            if not video:
                video_schedule = NoVideoSchedule()
            else:
                video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(
                self.env, log_dir, video_callable=video_schedule,
                force=True)
            self.monitoring = True

        self._log_dir = log_dir
        self._force_reset = force_reset

    @staticmethod
    def env_action_space_is_discrete(env):
        if isinstance(env.action_space, Discrete):
            return True
        elif isinstance(env.action_space, Box):
            return False
        else:
            raise NotImplementedError('Uknown base environment: ', env)

    @staticmethod
    def get_env_action_space_dim(env):
        if GymEnv.env_action_space_is_discrete(env):
            return env.action_space.n
        return env.action_space.shape[0]

    def reset(self):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __str__(self):
        return "GymEnv: %s" % self.env


class NormalizedGymEnv(Env):
    """Taken from rllab normalized_env.py
    """
    def __init__(self, env_name,
                 video=False,
                 log_dir=None,
                 force_reset=False,
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001):
        env = GymEnv(env_name, video, log_dir, force_reset)
        self.env = env
        self.env_id = env.spec.id
        self.spec = env.spec
        self.observation_space = env.observation_space
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(self.env.observation_space.shape)
        self._obs_var = np.ones(self.env.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def _update_obs_estimate(self, obs):
        flat_obs = obs
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean +\
            self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var +\
            self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean\
            + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var +\
            self._reward_alpha * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        # rllab does not subtract mean
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        ret = self.env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        return ret

    @property
    def action_space(self):
        if isinstance(self.env.action_space, Box):
            ub = np.ones(self.env.action_space.shape)
            return Box(-1 * ub, ub)
        return self.env.action_space

    def step(self, action):
        if isinstance(self.env.action_space, Box):
            # rescale the action
            lb, ub = self.env.action_space.low, self.env.action_space.high
            scaled_action = lb + (action[0] + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self.env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized GymEnv: %s" % self.env
