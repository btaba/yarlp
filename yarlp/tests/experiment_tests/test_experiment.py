import unittest
import os
import shutil

from yarlp.experiment.experiment import Experiment


class TestExperiment(unittest.TestCase):

    def setUp(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, 'test_experiment.json')
        self.e = Experiment(file, n_jobs=4)

    def test_run(self):
        self.e.run()
        self.assertTrue(os.path.isdir(self.e._experiment_dir))

    def tearDown(self):
        shutil.rmtree(os.path.dirname(self.e._experiment_dir))
