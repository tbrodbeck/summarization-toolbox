import environs
from evaluator.eval_util import ModelInfoReader
from timelogging.timeLog import log
import unittest

class Test_Model_Info_Reader(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(Test_Model_Info_Reader, self).__init__(*args, **kwargs)
    env = environs.Env()
    env.read_env()
    self.run_path = env('oneDriveDir') + 'studyProjectLuca/results/t5-de/0'
    self.run_path_with_checkpoint = self.run_path + '/checkpoint-5000'
    self.model_info_reader = ModelInfoReader(self.run_path)
    self.model_info_reader_with_checkpoints = ModelInfoReader(self.run_path_with_checkpoint)

  def test_read_model_info(self):
    assert type(self.model_info_reader.language) == str
    assert type(self.model_info_reader.model_name) == str
    assert type(self.model_info_reader.run_name) == str

    assert self.model_info_reader.language == self.model_info_reader_with_checkpoints.language
    assert self.model_info_reader.model_name == self.model_info_reader_with_checkpoints.model_name
    assert self.model_info_reader.run_name == self.model_info_reader_with_checkpoints.run_name
