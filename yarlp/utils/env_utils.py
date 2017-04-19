import gym
from gym.spaces import Discrete, Box
from gym.core import Env


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


# class FixedIntervalVideoSchedule(object):
#     def __init__(self, interval):
#         self.interval = interval

#     def __call__(self, count):
#         return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


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
