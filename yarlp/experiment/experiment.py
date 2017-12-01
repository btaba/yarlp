import os
import gym
import sys
import json
import copy
import click
import subprocess
import pandas as pd

from jsonschema import validate
from itertools import product
from yarlp.experiment import plotting
from concurrent.futures import ProcessPoolExecutor
from yarlp.experiment.experiment_schema import schema
from yarlp.experiment.job import Job
from yarlp.utils import experiment_utils


class Experiment(object):
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
    def from_json_spec(cls, json_spec_filename, log_dir=None, *args, **kwargs):
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
        assert json_spec_filename is not None
        assert os.path.exists(json_spec_filename) and\
            os.path.isfile(json_spec_filename)

        spec_file_handle = open(json_spec_filename, 'r')
        _raw_spec = json.load(spec_file_handle)

        # validate the json spec using jsonschema
        validate(_raw_spec, schema)

        # create a json spec for each env-agent-repeat
        # _spec_list = cls._spec_product(_raw_spec)
        _spec_list = _raw_spec['runs']

        cls._validate_agent_names(_spec_list)
        _spec_list = cls._add_validated_agent_repeats(_spec_list)

        # if the agent params are lists, then we need to split them up further
        # since these will be the cross-validation (cv) folds
        cls._spec_list = cls._expand_agent_grid(_spec_list)

        # create log directory and save the full spec to the directory
        if not log_dir:
            cls._experiment_dir = cls._create_log_dir(
                json_spec_filename)
        else:
            cls._experiment_dir = log_dir
        experiment_utils._save_spec_to_dir(cls._spec_list, cls._experiment_dir)
        return cls

    def run(self, n_jobs=None):
        if self.video:
            # GUI operations don't play nice with parallel execution
            for j in self._jobs:
                j()
        else:
            with ProcessPoolExecutor() as ex:
                ex.map(self.run_job, self._spec_list)

    def run_job(self, s):
        env = os.environ.copy()
        python_path = sys.executable
        command = ('{} -m yarlp.experiment.job --log-dir {}'
                   ' --video {} --spec \'{}\'').format(
            python_path, self._experiment_dir, self.video, json.dumps(s))
        p = subprocess.Popen(command, env=env, shell=True)
        out, err = p.communicate()
        return out, err

    @property
    def _jobs(self):
        for s in self._spec_list:
            yield Job(s, self._experiment_dir, self.video)

    @property
    def spec_list(self):
        return self._spec_list

    def _add_validated_agent_repeats(self, spec_list):
        """
        Validates the environment names and adds a json spec
        for each repeat with a 'run_name'
        """
        env_list = [x.id for x in gym.envs.registry.all()]

        repeated_spec_list = []
        for s in spec_list:
            env_name = s['env']['name']
            assert env_name in env_list,\
                "{} is not an available environment name.".format(env_name)

            for r in s['agent']['seeds']:
                s_copy = copy.deepcopy(s)
                run_name = '{}_{}_run{}'.format(
                    s['env']['name'], s['agent']['type'], r)
                s_copy['run_name'] = run_name
                s_copy['seed'] = r
                repeated_spec_list.append(s_copy)

        return repeated_spec_list

    def _expand_agent_grid(self, spec_list):
        """
        If the agent params are lists, we need to expand them
        into one run for each parameter grid...effectively doing a grid search
        over a list of parameters
        """

        grid_search = []

        for s in spec_list:

            # expand the grid of params
            params = s['agent'].get('params', {})
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
                new_s['agent']['param'] = g
                new_s['param_run'] = count
                param_name = '_param{}'.format(count)
                new_s['run_name'] = new_s['run_name'] + param_name
                grid_search.append(new_s)
                count += 1

        return grid_search

    def _validate_agent_names(self, spec):
        agent_set = set()
        cls_dict = experiment_utils._get_agent_cls_dict()
        for s in spec:
            agent_name = s['agent']['type']
            assert agent_name in cls_dict,\
                "{} is not an implemented agent. Select one of {}".format(
                    agent_name, cls_dict.keys())

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

        return experiment_utils._create_log_directory(
            experiment_name, experiment_dir)


def _merge_stats(experiment_dir):
    """
    Loop through all experiments, and write all the stats
    back to the base repository
    """
    statspath = os.path.join(experiment_dir, 'stats')
    if not os.path.exists(statspath):
        os.makedirs(statspath)
    agg_stats_file = os.path.join(statspath, 'merged_stats.tsv')

    stats_list = []
    for d in os.listdir(experiment_dir):
        base_path = os.path.join(experiment_dir, d)
        if not os.path.isdir(base_path) or d == 'stats':
            continue
        spec = open(os.path.join(base_path, 'spec.json'), 'r')
        spec = json.load(spec)

        f = os.path.join(base_path, 'stats.json.txt')
        with open(f, 'r') as f:
            stats = list(map(json.loads, f.readlines()))
        stats = pd.DataFrame(stats)

        stats['param_run'] = spec['param_run']
        stats['run_name'] = spec['run_name']
        stats['agent'] = spec['agent']['type']
        stats['env'] = spec['env']['name']
        stats['agent_params'] = str(spec['agent']['param'])
        stats['seed'] = spec['seed']
        stats_list.append(stats)

    assert len(stats_list) > 0, "No stats were found."
    stats = stats_list[0]
    for s in stats_list[1:]:
        stats = stats.append(s)

    with open(agg_stats_file, 'w') as f:
        stats.to_csv(f, index=False, header=True, sep='\t')

    return stats


def _merge_benchmark_stats(experiment_dir):
    """
    Merge stats from OpenAI benchmarks experiments
    """
    statspath = os.path.join(experiment_dir, 'stats')
    if not os.path.exists(statspath):
        os.makedirs(statspath)
    agg_stats_file = os.path.join(statspath, 'merged_stats.tsv')

    stats_list = []
    for d in os.listdir(experiment_dir):
        base_path = os.path.join(experiment_dir, d)
        if not os.path.isdir(base_path) or d == 'stats':
            continue

        stats = pd.read_csv(os.path.join(base_path, 'progress.csv'))

        f = os.path.join(base_path, '0.monitor.csv')
        with open(f, 'r') as f:
            spec = f.readline()
            spec = json.loads(spec[1:])

        stats['env'] = spec['env_id']
        stats['run_name'] = d

        stats_list.append(stats)

    assert len(stats_list) > 0, "No stats were found."
    stats = stats_list[0]
    for s in stats_list[1:]:
        stats = stats.append(s)

    with open(agg_stats_file, 'w') as f:
        stats.to_csv(f, index=False, header=True, sep='\t')

    return stats


def generate_plots_benchmark_vs_yarlp(yarlp_dir, benchmark_dir):
    training_stats = _merge_stats(yarlp_dir)
    benchmark_stats = _merge_benchmark_stats(benchmark_dir)
    for env in training_stats.env.unique():
        benchmark = benchmark_stats[benchmark_stats.env == env].rename(
            columns={'EpRewMean': 'avg_total_reward',
                     'TimestepsSoFar': 'timesteps_so_far',
                     'TimeElapsed': 'time_elapsed',
                     'env': 'env_id'})
        benchmark['name'] = 'benchmark'
        benchmark['episode'] = 0
        for run_name in benchmark.run_name.unique():
            benchmark.loc[benchmark.run_name == run_name, 'episode'] = list(
                range(benchmark[benchmark.run_name == run_name].shape[0]))

        training_stats['name'] = 'yarlp'
        yarlp = training_stats[training_stats.env == env]

        merged_data = pd.concat([benchmark, yarlp])

        fig = plotting.make_plots(merged_data, env)

        fig.savefig(
            os.path.join(
                yarlp_dir,
                '{}.png'.format(env)))


@click.command()
@click.option('--spec-file',
              default='./experiment_configs/reinforce_experiment.json',
              help=('Path to json file spec if continue=False'
                    ', else path to experiment'))
@click.option('--video', default=False, type=bool,
              help='Whether to record video or not')
@click.option('--n-jobs', default=1, type=int,
              help='number of cpu cores to use when running experiments')
def run_experiment(spec_file, video, n_jobs):
    e = Experiment.from_json_spec(spec_file, video=video)
    e.run(n_jobs=n_jobs)


@click.command()
@click.option(
    '--upload-dir',
    help='Path of openai gym session to upload')
def upload_to_openai(upload_dir):
    gym.scoreboard.api_key = os.environ.get('OPENAI_GYM_API_KEY', None)
    gym.upload(upload_dir)


@click.command()
@click.option('--benchmark-name', default='Mujoco1M')
@click.option('--agent', default='TRPOAgent')
def run_benchmark(benchmark_name, agent):
    SEEDS = list(range(1, 100))

    from yarlp.experiment.benchmarks import _BENCHMARKS
    benchmark_dict = dict(
        map(lambda x: (x[1]['name'], x[0]), enumerate(_BENCHMARKS)))
    assert benchmark_name in benchmark_dict
    benchmark_idx = benchmark_dict[benchmark_name]
    benchmark = _BENCHMARKS[benchmark_idx]

    # Make a master log directory
    experiment_dir = Experiment._get_experiment_dir(
        benchmark_name)
    base_log_path = experiment_utils._create_log_directory(
        benchmark_name, experiment_dir)

    # write the json config for this baseline
    j = []
    for t in benchmark['tasks']:
        d = {
            "env": {
                "name": t['env_id'],
                "normalize_obs": True
            },
            "agent": {
                "type": agent,
                "seeds": SEEDS[:t['trials']],
                "training_params": {
                    "max_timesteps": t['num_timesteps']
                }
            }
        }
        j.append(d)

    j = {"runs": j}
    spec_file = os.path.join(base_log_path, 'spec.json')
    json.dump(j, open(spec_file, 'w'))

    # run the experiment
    e = Experiment.from_json_spec(
        spec_file, log_dir=base_log_path)
    e.run()


@click.command()
@click.argument('yarlp-dir')
@click.argument('openai-benchmark-dir')
def compare_benchmark(yarlp_dir, openai_benchmark_dir):
    generate_plots_benchmark_vs_yarlp(yarlp_dir, openai_benchmark_dir)
