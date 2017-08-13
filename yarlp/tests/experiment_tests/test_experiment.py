# import unittest
# import os
# import shutil

# from yarlp.experiment.experiment import Experiment


# class TestExperiment(unittest.TestCase):

#     def setUp(self):
#         dirname = os.path.dirname(os.path.abspath(__file__))
#         file = os.path.join(dirname, 'test_experiment.json')
#         print(file)
#         self.e = Experiment.from_json_spec(file)

#     def test_run(self):
#         self.e.run()
#         print(self.e._experiment_dir)
#         self.assertTrue(os.path.isdir(self.e._experiment_dir))

#     def tearDown(self):
#         shutil.rmtree(os.path.dirname(self.e._experiment_dir))
