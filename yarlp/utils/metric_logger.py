import os
import re
import json
import time
import logging
import numpy as np
from tabulate import tabulate
from collections import deque


class MetricLogger:
    """Logs metrics to console and to file
    """

    def __init__(self, log_dir=None, logger_name='yarlp'):
        self._log_dir = log_dir
        self._logger = self._create_logger(logger_name)
        self._metric_dict = {'Iteration': 0}
        self._iteration = 0

        if self._log_dir is not None:
            self._stat_file = os.path.join(self._log_dir, 'stats.json.txt')

        self._start_time = time.time()
        self._running_reward = deque(maxlen=40)

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
        self._iteration += 1
        self._metric_dict = {'Iteration': self._iteration}

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

        self['avg_episode_length'] = np.mean(rollout['episode_lengths'])
        self['episodes_this_iter'] = len(rollout['episode_returns'])
        self['timesteps_this_iter'] = len(rollout['dones'])
        self['min_reward'] = np.min(rollout['episode_returns'])
        self['max_reward'] = np.max(rollout['episode_returns'])
        self['training'] = train
        self['avg_total_reward'] = np.mean(rollout['episode_returns'])
        self._running_reward.extend(rollout['episode_returns'])
        self['Smoothed_total_reward'] = np.mean(self._running_reward)
        self['std_reward'] = np.std(rollout['episode_returns'])
        self['total_reward'] = np.sum(rollout['episode_returns'])
        self['time_elapsed'] = t
