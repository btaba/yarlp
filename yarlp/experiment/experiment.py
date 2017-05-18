import os
import gym
import json
import copy
import pandas as pd

import tqdm
from jsonschema import validate
from itertools import product
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from yarlp.experiment.experiment_schema import schema
from yarlp.experiment.job import Job
from yarlp.experiment.experiment_utils import ExperimentUtils


class Experiment(ExperimentUtils):
    def __init__(self, video=False):
        """
        Params
        ----------
        video (bool): False disables video recording. otherwise
            we us the defaul openai gym behavior of taking videos on
            every cube up until 1000, and then for every 1000 episodes.
        """
        self.video = video

    @classmethod
    def from_json_spec(cls, json_spec_filename, *args, **kwargs):
        """
        Reads in json_spec_filen of experiment, validates the experiment spec,
        creates a spec for each combination of agent/env/grid-search-params
        and creates the experiment directory
        Params
        ----------
        json_spec_filename (str): the file path of the json spec file for the
            complete experiment
        """
        cls = cls(*args, **kwargs)
        assert os.path.exists(json_spec_filename) and\
            os.path.isfile(json_spec_filename)

        _spec_filename = json_spec_filename
        spec_file_handle = open(json_spec_filename, 'r')
        _raw_spec = json.load(spec_file_handle)

        # validate the json spec using jsonschema
        validate(_raw_spec, schema)

        # create a json spec for each env-agent-repeat
        _spec_list = cls._spec_product(_raw_spec)

        cls._validate_agent_names(_spec_list)
        _spec_list = cls._add_validated_env_repeats(_spec_list)

        # if the agent params are lists, then we need to split them up further
        # since these will be the cross-validation (cv) folds
        cls._spec_list = cls._expand_agent_grid(_spec_list)

        # create log directory and save the full spec to the directory
        cls._experiment_dir = cls._create_log_dir(
            _spec_filename)
        Experiment._save_spec_to_dir(cls._spec_list, cls._experiment_dir)
        return cls

    @classmethod
    def from_unfinished_experiment_dir(cls, path, *args, **kwargs):
        """
        Restart experiment from unfinished experiment spec, this assumes that
        Experiment._save_spec_to_dir was run previously
        """
        cls = cls(*args, **kwargs)
        if not os.path.isdir(path):
            path = cls._get_experiment_dir(path)
        assert os.path.isdir(path),\
            '{} does not exist or is not a directory'.format(path)
        cls._experiment_dir = path

        # get the spec file
        spec_file_path = os.path.join(path, 'spec.json')
        assert os.path.isfile(spec_file_path)
        cls._spec_list = json.load(open(spec_file_path, 'r'))
        assert isinstance(cls._spec_list, list)

        # walk the file path and dequeue specs that were completed
        path_walk = [p[0] for p in os.walk(path)
                     if os.path.isdir(p[0]) and p[0] != path]
        completed_specs = []
        for p in path_walk:
            # did the experiment finish?
            stat_file = os.path.join(p, 'stats.csv')
            if not os.path.exists(stat_file):
                continue

            # check stat_file to see if experiment was run to completion
            spec_file_path = os.path.join(p, 'spec.json')
            spec_dict = json.load(open(spec_file_path, 'r'))
            stat_df = pd.read_csv(stat_file)

            num_training_epochs = spec_dict['agents']['training_epochs']
            if (stat_df.training == True).sum() < num_training_epochs:
                continue

            completed_specs.append(spec_dict)

        # dequeue complete experiments
        cls._spec_list = [sl for sl in cls._spec_list
                          if sl not in completed_specs]

        return cls

    def run(self, n_jobs=1):
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
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
        for s in tqdm.tqdm(self._spec_list):
            yield Job(s, self._experiment_dir, self.video)

    @property
    def spec_list(self):
        return self._spec_list

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

    def _expand_agent_grid(self, spec_list):
        """If the agent params are lists, we need to expand them
        into one run for each parameter grid...effectively doing a grid search
        over a list of parameters
        """

        grid_search = []

        for s in spec_list:

            # expand the grid of params
            params = s['agents']['params']
            singleton_params = {
                k: v for k, v in params.items() if not isinstance(v, list)}
            grid_params = {
                k: v for k, v in params.items() if isinstance(v, list)}
            grid_params = [
                dict(zip(grid_params.keys(), x))
                for x in product(*grid_params.values())]
            grid_params = [{**g, **singleton_params} for g in grid_params]

            # add a spec with each agent param in the grid
            count = 0
            for g in grid_params:
                new_s = copy.deepcopy(s)
                new_s['agents']['params'] = g
                new_s['param_run'] = count
                param_name = '_param{}'.format(count)
                new_s['run_name'] = new_s['run_name'] + param_name
                grid_search.append(new_s)
                count += 1

        return grid_search

    def _validate_agent_names(self, spec):
        agent_set = set()
        cls_dict = Experiment._get_agent_cls_dict()
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

    @staticmethod
    def _get_experiment_dir(experiment_name):
        home = os.path.expanduser('~')
        experiment_dir = os.path.join(
            home, 'yarlp_experiments', experiment_name)
        return experiment_dir

    def _create_log_dir(self, spec_filename):
        base_filename = os.path.basename(spec_filename)
        experiment_name = base_filename.split('.')[0]

        experiment_dir = Experiment._get_experiment_dir(
            experiment_name)

        return Experiment._create_log_directory(
            experiment_name, experiment_dir)

    def _merge_stats(self):
        """Loop through all experiments, and write all the stats
        back to the base repository
        """
        statspath = os.path.join(self._experiment_dir, 'stats')
        if not os.path.exists(statspath):
            os.makedirs(statspath)
        agg_stats_file = os.path.join(statspath, 'merged_stats.csv')

        for d in os.listdir(self._experiment_dir):
            base_path = os.path.join(self._experiment_dir, d)
            if not os.path.isdir(base_path) or d == 'stats':
                continue
            spec = open(os.path.join(base_path, 'spec.json'), 'r')
            spec = json.load(spec)
            stats = pd.read_csv(os.path.join(base_path, 'stats.csv'))
            stats['run'] = spec['run']
            stats['param_run'] = spec['param_run']
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
        """Compute aggregates on the merged_stats.csv file
        grouped by [agent, env, training, episode, agent_params]
        so we average over several runs
        """
        m = os.path.join(self._experiment_dir, 'stats/merged_stats.csv')
        assert os.path.exists(m), "Merged stats file must exist"
        agg_episode_stats_file = os.path.join(
            self._experiment_dir, 'stats/agg_episode_stats.csv')
        agg_agent_stats_file = os.path.join(
            self._experiment_dir, 'stats/agg_agent_stats.csv')
        base_df = pd.read_csv(m)

        # episode level stats
        df_avg = base_df.groupby(
            ['agent', 'env', 'training', 'episode', 'agent_params']
        ).mean().add_prefix('mean_')
        df_std = base_df.groupby(
            ['agent', 'env', 'training', 'episode', 'agent_params']
        )['avg_episode_length', 'total_reward', 'total_episode_length']
        df_std = df_std.std().add_prefix('std_').fillna(0)
        df = df_avg.join(df_std)
        df.to_csv(agg_episode_stats_file)

        # agent level stats
        df = base_df.groupby(
            ['agent', 'env', 'training', 'agent_params', 'run']
        ).apply(lambda x: x.sort_values('episode').mean())
        df = df[
            ['avg_episode_length', 'total_reward', 'total_episode_length']]
        df = df.reset_index()
        df_avg = df.groupby(
            ['agent', 'env', 'training', 'agent_params']).mean()
        df_std = df.groupby(['agent', 'env', 'training', 'agent_params']).std()
        df_std = df_std.add_prefix('std_').fillna(0)
        df = df_avg.join(df_std)
        df.drop(['run', 'std_run'], inplace=True, axis=1)
        df.to_csv(agg_agent_stats_file, index=True)

    def _plot_stats(self):
        """Take the agg_stats.csv file and make plots
        """
        m = os.path.join(self._experiment_dir, 'stats/agg_episode_stats.csv')
        assert os.path.exists(m), "Aggregated stats file must exist"

        df = pd.read_csv(m)
        plt_index = df.groupby(
            ['agent', 'env', 'training', 'agent_params']).count().index

        for count, idx in enumerate(plt_index):
            sub_df = df[
                (df.agent == idx[0]) & (df.env == idx[1]) &
                (df.training == idx[2]) & (df.agent_params == idx[3])].copy()
            sub_df.sort_values('episode', inplace=True)
            sub_df.reset_index(inplace=True, drop=True)

            plt.figure(figsize=(8, 14))

            plt.subplot(4, 1, 1)
            plt.errorbar(
                sub_df.mean_total_episode_length.cumsum(),
                sub_df.mean_total_reward,
                sub_df.std_total_reward, sub_df.std_total_episode_length)
            plt.title(idx)
            plt.ylabel('Total Reward')
            plt.xlabel('Steps')

            plt.subplot(4, 1, 2)
            plt.errorbar(sub_df.episode, sub_df.mean_total_reward,
                         sub_df.std_total_reward)
            plt.ylabel('Total Reward')
            plt.xlabel('Episodes')

            plt.subplot(4, 1, 3)
            plt.errorbar(
                sub_df.mean_total_episode_length.cumsum(),
                sub_df.mean_avg_episode_length,
                sub_df.std_avg_episode_length, sub_df.std_total_episode_length)
            plt.ylabel('Episode Length')
            plt.xlabel('Steps')

            plt.subplot(4, 1, 4)
            plt.errorbar(
                sub_df.episode, sub_df.mean_avg_episode_length,
                sub_df.std_avg_episode_length)
            plt.ylabel('Episode Length')
            plt.xlabel('Episodes')

            png_str = '_'.join(idx[:2])
            png_str += '_{}_'.format(count)
            png_str += '_train.png' if idx[2] else '_test.png'
            png_path = os.path.join(self._experiment_dir, 'stats', png_str)
            plt.savefig(png_path)
            plt.close()
