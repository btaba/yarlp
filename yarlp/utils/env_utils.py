from gym.spaces import Discrete, Box


def env_action_space_is_discrete(env):
    if isinstance(env.action_space, Discrete):
        return True
    elif isinstance(env.action_space, Box):
        return False
    else:
        raise NotImplementedError('Uknown base environment: ', env)


def get_env_action_space_dimension(env):
    if env_action_space_is_discrete(env):
        return env.action_space.n
    return env.action_space.shape[0]
