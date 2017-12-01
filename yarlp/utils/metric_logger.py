import os
import re
import json
import time
import logging
import numpy as np

from tabulate import tabulate
from yarlp.utils.replay_buffer import Rollout


class MetricLogger:
    """Logs metrics to console and to file
    """

    def __init__(self, log_dir=None, logger_name='yarlp'):
        self._log_dir = log_dir
        self._logger = self._create_logger(logger_name)
        self._metric_dict = {'episode': 0}
        self._episode = 0

        if self._log_dir is not None:
            self._stat_file = os.path.join(self._log_dir, 'stats.json.txt')

        self._start_time = time.time()

    def __setitem__(self, metric_name, value):
        self._validate_header_name(metric_name)
        self._metric_dict[metric_name] = value

    def __getitem__(self, metric_name):
        return self._metric_dict[metric_name]

    @property
    def metric_dict(self):
        for k, v in self._metric_dict.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                self._metric_dict[k] = float(v)
        return self._metric_dict

    def add_metric(self, metric_name, value):
        self[metric_name] = value

    def _create_logger(self, name):
        logger = logging.getLogger(name)

        # remove old handlers
        for handler in logger.handlers:  # remove all old handlers
            logger.removeHandler(handler)

        # add new handlers
        handlers = [logging.StreamHandler()]
        if self._log_dir:
            log_file = os.path.join(self._log_dir, 'logs.log')
            handlers.append(logging.FileHandler(log_file))

        for handler in handlers:
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
        self._metric_dict = {'episode': self._episode}

    def _tabulate(self):
        tabulate_list = list(
            zip(list(self._metric_dict),
                self._metric_dict.values()))
        tabulate_list = sorted(tabulate_list, key=lambda x: x[0])
        return tabulate(tabulate_list, floatfmt=".4f")

    def log(self):
        """
        Log to file in the log directory
        """
        self._logger.info(self._tabulate())

        if self._log_dir is not None:
            self._logger.info('Writing stats to {}'.format(self._stat_file))
            with open(self._stat_file, 'a') as f:
                json.dump(self.metric_dict, f)
                f.write('\n')

        self._reset_metrics()

    def set_metrics_for_rollout(self, rollout, train=True):
        t = round(time.time() - self._start_time, 6)
        if isinstance(rollout, list):
            # unroll the rollout
            self['avg_episode_length'] = np.mean(
                [len(r.rewards) for r in rollout])
            self['episodes_this_iter'] = len(rollout)
            self['timesteps_this_iter'] = np.sum(
                [len(r.rewards) for r in rollout])
            self['min_reward'] = np.min([np.sum(r.rewards) for r in rollout])
            self['max_reward'] = np.max([np.sum(r.rewards) for r in rollout])
            self['training'] = train
            self['avg_reward'] = np.mean([np.mean(r.rewards) for r in rollout])
            self['avg_total_reward'] = np.mean(
                [np.sum(r.rewards) for r in rollout])
            self['std_reward'] = np.std([np.sum(r.rewards) for r in rollout])
            self['total_reward'] = np.sum([np.sum(r.rewards) for r in rollout])
            self['time_elapsed'] = t
        else:
            assert isinstance(rollout, Rollout)
            self['avg_episode_length'] = len(rollout.rewards)
            self['episodes_this_iter'] = 1
            self['timesteps_this_iter'] = len(rollout.rewards)
            self['min_reward'] = np.min(rollout.rewards)
            self['max_reward'] = np.max(rollout.rewards)
            self['training'] = train
            self['avg_reward'] = np.mean(rollout.rewards)
            self['avg_total_reward'] = np.sum(rollout.rewards)
            self['std_reward'] = np.std(rollout.rewards)
            self['total_reward'] = np.sum(rollout.rewards)
            self['time_elapsed'] = t
