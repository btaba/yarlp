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

    def __init__(self, log_dir=None, logger_name='yarlp', reward_len=100):
        self._log_dir = log_dir
        self._logger_name = logger_name
        self._metric_dict = {'Iteration': 0}
        self._iteration = 0
        self._logger = self.set_logger()

        if self._log_dir is not None:
            self._stat_file = os.path.join(self._log_dir, 'stats.json.txt')

        self._start_time = time.time()
        self._running_reward = deque(maxlen=reward_len)

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

    @property
    def logger(self):
        return self._logger

    def set_logger(self):
        logger = logging.getLogger(self._logger_name)

        # remove old handlers
        for handler in list(logger.handlers):  # remove all old handlers
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
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        self.logger.info('\n' + self._tabulate())

        if self._log_dir is not None:
            self.logger.info('Writing stats to {}'.format(self._stat_file))
            with open(self._stat_file, 'a') as f:
                json.dump(self.metric_dict, f)
                f.write('\n')

        self._reset_metrics()

    def set_metrics_for_iter(self, episode_returns):
        t = round(time.time() - self._start_time, 6)
        self['episodes_this_iter'] = len(episode_returns)
        self['min_reward'] = min(episode_returns)
        self['max_reward'] = max(episode_returns)
        self['avg_total_reward'] = np.mean(episode_returns)
        [self._running_reward.append(r) for r in episode_returns]
        self['Smoothed_total_reward'] = np.mean(self._running_reward)
        self['std_reward'] = np.std(episode_returns)
        self['time_elapsed'] = t

    def set_metrics_for_rollout(self, rollout, train=True):
        t = round(time.time() - self._start_time, 6)
        if type(rollout) is list:
            self['avg_episode_length'] = np.mean([np.mean(r['episode_lengths']) for r in rollout])
            self['episodes_this_iter'] = sum([len(r['episode_returns']) for r in rollout])
            self['timesteps_this_iter'] = sum([len(r['dones']) for r in rollout])
            self['min_reward'] = np.min([np.min(r['episode_returns']) for r in rollout])
            self['max_reward'] = np.max([np.max(r['episode_returns']) for r in rollout])
            self['training'] = train
            self['avg_total_reward'] = np.mean([np.mean(r['episode_returns']) for r in rollout])
            [self._running_reward.extend(r['episode_returns']) for r in rollout]
            self['Smoothed_total_reward'] = np.mean(self._running_reward)
            self['std_reward'] = np.mean([np.std(r['episode_returns']) for r in rollout])
            self['total_reward'] = np.mean([np.sum(r['episode_returns']) for r in rollout])
            self['time_elapsed'] = t
        else:
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


def explained_variance(y, pred):
    explained_variance = np.nan if np.var(y) == 0\
        else 1 - np.var(y - pred) /\
        np.var(y)
    return explained_variance
