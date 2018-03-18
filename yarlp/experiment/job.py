import os
import json
import click
import traceback
import tensorflow as tf
from yarlp.utils import experiment_utils
from yarlp.utils.env_utils import NormalizedGymEnv, ParallelEnvs


class Job(object):
    def __init__(self, spec_dict, log_dir, video, load_agent=False):
        self._spec_dict = spec_dict
        self._log_dir = log_dir
        self._video = video
        self._load_agent = load_agent

    def _load(self):
        existing_dirs = [d for d in os.listdir(self._log_dir)
                         if d.startswith(self._spec_dict['run_name'])]
        if self._load_agent and len(existing_dirs) > 0:
            # load the agent from a saved directory if it exists
            self._job_dir = os.path.join(self._log_dir, existing_dirs[0])
            self._agent = self._get_agent(load=True)
            self._env = self._agent.env
        else:
            # start fresh
            self._job_dir = self._create_log_dir()
            self._env = self._get_env(self._job_dir)
            self._agent = self._get_agent()

    def __call__(self):
        self._load()
        training_params = self._spec_dict['agent'].get('training_params', {})
        try:
            self._agent.train(**training_params)
        except Exception as e:
            traceback.print_exc()
        self._env.close()
        tf.reset_default_graph()

    def _get_env(self, job_dir):
        env_name = self._spec_dict['env']['name']
        kwargs = {k: v for k, v in self._spec_dict['env'].items()
                  if k != 'name' and k != 'is_parallel'}
        kwargs['video'] = self._video
        if not self._spec_dict['env'].get('is_parallel', False):
            env = NormalizedGymEnv(
                env_name, log_dir=job_dir, force_reset=True,
                **kwargs)
        else:
            env = ParallelEnvs(
                env_name, log_dir=job_dir, force_reset=True, **kwargs)
        if 'timestep_limit' in self._spec_dict['env']:
            env.spec.timestep_limit = self._spec_dict['env']['timestep_limit']
        return env

    def _get_agent(self, load=False):
        cls_dict = experiment_utils._get_agent_cls_dict()
        params = self._spec_dict['agent'].get('params', {})
        params['seed'] = self._spec_dict['seed']
        agent_cls = cls_dict[self._spec_dict['agent']['type']]
        if load:
            return agent_cls.load(self._job_dir)
        return agent_cls(env=self._env, log_dir=self._job_dir, **params)

    def _create_log_dir(self):
        dir_name = self._spec_dict['run_name']
        job_dir = experiment_utils._create_log_directory(
            dir_name, self._log_dir)
        experiment_utils._save_spec_to_dir(
            {"runs": self._spec_dict}, job_dir)
        return job_dir


@click.command()
@click.option('--spec', type=str)
@click.option('--log-dir', type=str)
@click.option('--video', type=bool, default=False)
@click.option('--reload-exp', type=bool, default=False)
def run_job(spec, log_dir, video, reload_exp):
    j = Job(json.loads(spec), log_dir, video, reload_exp)
    j()


if __name__ == '__main__':
    run_job()
