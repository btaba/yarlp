import os
import re
import csv
import logging
import numpy as np

from tabulate import tabulate
from yarlp.utils.replay_buffer import Rollout


class MetricLogger:
    """Logs metrics to console and to file
    """

    def __init__(self, log_dir=None, logger_name='yarlp'):
        # assert isinstance(log_dir, str)
        # assert isinstance(logger_name, str)

        self._log_dir = log_dir
        self._logger = self._create_logger(logger_name)
        self._metric_dict = {'episode': [0]}
        self._episode = 0

        if self._log_dir is not None:
            self._stat_file = os.path.join(self._log_dir, 'stats.csv')

    def __setitem__(self, metric_name, value):
        self._validate_header_name(metric_name)
        self._metric_dict[metric_name] = [value]

    def __getitem__(self, metric_name):
        return self._metric_dict[metric_name]

    def add_metric(self, metric_name, value):
        self[metric_name] = value

    def _create_logger(self, name):
        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(20)
        logger.propagate = False
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        return logger

    def _validate_header_name(self, header_name):
        assert re.match('^[A-Za-z_-]+$', header_name) is not None,\
            ("Metric name '{}'".format(header_name),
             "must have only ascii letters and _ or -")

    def _reset_metrics(self):
        self._episode += 1
        self._metric_dict = {'episode': [self._episode]}

    def _tabulate(self):
        tabulate_list = list(
            zip(list(self._metric_dict),
                self._metric_dict.values()))
        return tabulate(tabulate_list, floatfmt=".4f")

    def log(self, reset=True):
        """Log to csv in the log directory
        """
        self._logger.info(self._tabulate())

        if self._log_dir is not None:
            self._logger.info('Writing stats to {}'.format(self._stat_file))
            if not os.path.isfile(self._stat_file):
                with open(self._stat_file, 'w') as f:
                    w = csv.writer(f)
                    w.writerow(self._metric_dict.keys())
            with open(self._stat_file, 'a') as f:
                w = csv.writer(f)
                vals = [v[0] for v in self._metric_dict.values()]
                w.writerow(vals)

        if reset:
            self._reset_metrics()

    def set_metrics_for_rollout(self, rollout, train=True):

        if isinstance(rollout, list):
            # unroll the rollout
            self['episode_length'] = np.mean([len(r.rewards) for r in rollout])
            self['training'] = train
            self['avg_reward'] = np.mean([np.mean(r.rewards) for r in rollout])
            self['avg_total_reward'] = np.mean(
                [np.sum(r.rewards) for r in rollout])
            self['std_reward'] = np.std([np.std(r.rewards) for r in rollout])
            self['total_reward'] = np.sum([np.sum(r.rewards) for r in rollout])
        else:
            assert isinstance(rollout, Rollout)
            self['episode_length'] = len(rollout.rewards)
            self['training'] = train
            self['avg_reward'] = np.mean(rollout.rewards)
            self['avg_total_reward'] = np.sum(rollout.rewards)
            self['std_reward'] = np.std(rollout.rewards)
            self['total_reward'] = np.sum(rollout.rewards)
