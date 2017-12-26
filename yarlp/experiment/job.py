import json
import click
import tensorflow as tf
from yarlp.utils import experiment_utils
from yarlp.utils.env_utils import NormalizedGymEnv
from yarlp.utils.metric_logger import MetricLogger


class Job(object):
    def __init__(self, spec_dict, log_dir, video):
        self._spec_dict = spec_dict
        self._log_dir = log_dir
        self._video = video

    def _load(self):
        self._job_dir = self._create_log_dir()
        self._env = self._get_env(self._job_dir)
        self._agent = self._get_agent()

    def __call__(self):
        self._load()
        training_params = self._spec_dict['agent'].get('training_params', {})
        try:
            self._agent.train(**training_params)
        except Exception as e:
            print(e)
        self._env.close()
        tf.reset_default_graph()

    def _get_env(self, job_dir):
        env_name = self._spec_dict['env']['name']
        env = NormalizedGymEnv(
            env_name, self._video, job_dir, force_reset=True,
            **self._spec_dict['env'])
        if 'timestep_limit' in self._spec_dict['env']:
            env.spec.timestep_limit = self._spec_dict['env']['timestep_limit']
        return env

    def _get_agent(self):
        cls_dict = experiment_utils._get_agent_cls_dict()
        params = self._spec_dict['agent'].get('params', {})
        params['seed'] = self._spec_dict['seed']
        agent_cls = cls_dict[self._spec_dict['agent']['type']]
        metric_logger = MetricLogger(self._job_dir)
        return agent_cls(env=self._env, logger=metric_logger, **params)

    def _create_log_dir(self):
        dir_name = self._spec_dict['run_name']
        job_dir = experiment_utils._create_log_directory(
            dir_name, self._log_dir)
        experiment_utils._save_spec_to_dir(self._spec_dict, job_dir)
        return job_dir


@click.command()
@click.option('--spec', type=str)
@click.option('--log-dir', type=str)
@click.option('--video', type=bool, default=False)
def run_job(spec, log_dir, video):
    j = Job(json.loads(spec), log_dir, video)
    j()


if __name__ == '__main__':
    run_job()
