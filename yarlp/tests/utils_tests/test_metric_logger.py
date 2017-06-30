import unittest
import os
from yarlp.utils.metric_logger import MetricLogger


class TestMetricLogger(unittest.TestCase):

    def setUp(self):
        logger = MetricLogger(log_dir='.', logger_name='testytest')
        self.logger = logger

    def test_logger(self):
        self.logger.add_metric('a', 1)
        self.logger.log()

        self.logger.add_metric('a', 2)
        self.assertEqual(self.logger._episode, 1)

        # check that csv was created
        self.assertTrue(os.path.exists('stats.tsv'))

    def tearDown(self):
        os.remove('stats.tsv')
