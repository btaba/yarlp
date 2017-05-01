import os
import gym
import json
import copy
import pandas as pd

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

        # aggregate the stats across all jobs
        self._merge_stats()
        self._aggregate_stats_over_runs()
        self._plot_stats()

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
                s_copy['run'] = r
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

    def _merge_stats(self):
        agg_stats_file = os.path.join(self._experiment_dir, 'merged_stats.csv')

        for d in os.listdir(self._experiment_dir):
            base_path = os.path.join(self._experiment_dir, d)
            if not os.path.isdir(base_path):
                continue
            spec = open(os.path.join(base_path, 'spec.json'), 'r')
            spec = json.load(spec)
            stats = pd.read_csv(os.path.join(base_path, 'stats.csv'))
            stats['run'] = spec['run']
            stats['run_name'] = spec['run_name']
            stats['agent'] = spec['agents']['type']
            stats['env'] = spec['envs']['name']
            stats['env_timestep_limit'] = spec['envs']['timestep_limit']
            stats['env_normalize_obs'] = spec['envs']['normalize_obs']
            stats['agent_params'] = str(spec['agents']['params'])

            header = False
            if not os.path.exists(agg_stats_file):
                header = True

            with open(agg_stats_file, 'a') as f:
                stats.to_csv(f, index=False, header=header)

    def _aggregate_stats_over_runs(self):
        m = os.path.join(self._experiment_dir, 'merged_stats.csv')
        assert os.path.exists(m), "Merged stats file must exist"
        agg_stats_file = os.path.join(self._experiment_dir, 'agg_stats.csv')
        base_df = pd.read_csv(m)
        df_avg = base_df.groupby(
            ['agent', 'env', 'training', 'episode', 'agent_params']
        ).mean().add_prefix('mean_')
        df_std = base_df.groupby(
            ['agent', 'env', 'training', 'episode', 'agent_params']
        ).std().add_prefix('std_').fillna(0)
        df = df_avg.join(df_std)
        df.to_csv(agg_stats_file)

    def _plot_stats(self):
        pass
