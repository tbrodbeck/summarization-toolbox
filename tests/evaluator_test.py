import environs
import evaluator.main as o
import unittest

class TestDataProvider(unittest.TestCase):
  def test_provideData(self):
    env = environs.Env()
    env.read_env()
    o.evaluate(env('oneDriveDir') + 'studyProjectLuca/results/t5-de-new/long_run')
