import gym
from yarlp.experiment.experiment_utils import ExperimentUtils


class Job(ExperimentUtils):
    def __init__(self, spec_dict, log_dir):
        self._spec_dict = spec_dict
        self._log_dir = log_dir
        self._load()

    def _load(self):
        self._env = self._get_env()
        self._agent = self._get_agent()
        self._job_dir = self._create_log_dir()
        self._training_epochs = self._spec_dict['agents']['training_epochs']
        self._testing_epochs = self._spec_dict['agents']['testing_epochs']

    def __call__(self):
        self._agent.train(self._training_epochs, self._testing_epochs)
        self._env.close()

    def _get_env(self):
        env_name = self._spec_dict['envs']['name']
        env = gym.make(env_name)

        if 'timestep_limit' in self._spec_dict['envs']:
            env.spec.timestep_limit = self._spec_dict['envs']['timestep_limit']

        return env

    def _get_agent(self):
        cls_dict = Job.get_agent_cls_dict()
        params = self._spec_dict['agents']['params']
        agent_cls = cls_dict[self._spec_dict['agents']['type']]
        return agent_cls(env=self._env, **params)

    def _create_log_dir(self):
        dir_name = self._spec_dict['envs']['run_name']
        return Job.create_log_directory(dir_name, self._log_dir)
