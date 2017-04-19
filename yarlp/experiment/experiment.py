import os
import gym
import json
import copy

from jsonschema import validate
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from yarlp.experiment.experiment_schema import schema
from yarlp.experiment.job import Job
from yarlp.experiment.experiment_utils import ExperimentUtils


class Experiment(ExperimentUtils):
    def __init__(self, json_spec_filename, n_jobs=1,
                 video=False):
        """

        Params
        ----------
        video (bool): False disables video recording. otherwise
            we us the defaul openai gym behavior of taking videos on
            every cube up until 1000, and then for every 1000 episodes.
        """
        assert os.path.exists(json_spec_filename) and\
            os.path.isfile(json_spec_filename)

        self._spec_filename = json_spec_filename
        spec_file_handle = open(json_spec_filename, 'r')
        self._raw_spec = json.load(spec_file_handle)
        self.n_jobs = n_jobs
        self.video = video

        # validate the json spec using jsonschema
        validate(self._raw_spec, schema)

        # create a json spec for each env-agent-repeat
        _spec_list = self._spec_product(self._raw_spec)
        self._validate_agent_names(_spec_list)
        self._spec_list = self._add_validated_env_repeats(_spec_list)

        # create log directory and save the full spec to the directory
        self._experiment_dir = self._create_log_dir(
            self._spec_filename)
        Experiment.save_spec_to_dir(self._spec_list, self._experiment_dir)

    def run(self):
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            for j in self._jobs:
                future = ex.submit(j)
                res = future.result()
                if res:
                    print(res)

    @property
    def _jobs(self):
        for s in self._spec_list:
            yield Job(s, self._experiment_dir, self.video)

    def _spec_product(self, spec):
        """Cartesian product of a json spec, list of specs unique by env and agent
        """
        keys = spec.keys()
        spec_list = [spec[k] for k in keys]
        spec_product = product(*spec_list)
        spec_dicts = [
            {k: s for k, s in zip(keys, sp)}
            for sp in spec_product
        ]
        return spec_dicts

    def _add_validated_env_repeats(self, spec_list):
        """Validates the environment names and adds a json spec
        for each repeat with a 'run_name'
        """
        env_list = [x.id for x in gym.envs.registry.all()]

        repeated_spec_list = []
        for s in spec_list:
            env_name = s['envs']['name']
            assert env_name in env_list,\
                "{} is not an available environment name.".format(env_name)

            for r in range(s['envs']['repeats']):
                s_copy = copy.deepcopy(s)
                run_name = '{}_{}_run{}'.format(
                    s['envs']['name'], s['agents']['type'], r)
                s_copy['run_name'] = run_name
                repeated_spec_list.append(s_copy)

        return repeated_spec_list

    def _validate_agent_names(self, spec):
        agent_set = set()
        cls_dict = Experiment.get_agent_cls_dict()
        for s in spec:
            agent_name = s['agents']['type']
            assert agent_name in cls_dict,\
                "{} is not an implemented agent. Select one of {}".format(
                    agent_name, cls_dict.keys())

            assert agent_name not in agent_set,\
                """{} is duplicated in the experiment spec,
                please remove it""".format(
                    agent_name)

            agent_set.add(agent_name)

    def _create_log_dir(self, spec_filename):
        base_filename = os.path.basename(spec_filename)
        experiment_name = base_filename.split('.')[0]

        home = os.path.expanduser('~')
        experiment_dir = os.path.join(
            home, 'yarlp_experiments', experiment_name)

        return Experiment.create_log_directory(
            experiment_name, experiment_dir)
