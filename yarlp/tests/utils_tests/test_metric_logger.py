import pytest
import os
from yarlp.utils.metric_logger import MetricLogger


@pytest.fixture
def get_logger():
    logger = MetricLogger(log_dir='.', logger_name='testytest')
    yield logger
    os.remove('stats.json.txt')


def test_logger(get_logger):
    logger = get_logger
    logger.add_metric('a', 1)
    logger.log()

    logger.add_metric('a', 2)
    assert logger._iteration == 1

    # check that csv was created
    assert os.path.exists('stats.json.txt') is True
