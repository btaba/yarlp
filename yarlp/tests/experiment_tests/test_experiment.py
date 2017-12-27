import pytest
import os
import shutil

from yarlp.experiment.experiment import Experiment


@pytest.fixture
def get_exp():
    dirname = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dirname, 'test_experiment.json')
    print(file)
    e = Experiment.from_json_spec(file)
    yield e
    shutil.rmtree(os.path.dirname(e._experiment_dir))


def test_run(get_exp):
    e = get_exp
    e.run()
    print(e._experiment_dir)
    assert os.path.isdir(e._experiment_dir) is True
