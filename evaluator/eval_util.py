"""
General utilities used for the evaluation
"""
import os
from typing import List
import yaml

class ModelInfoReader():
  def __init__(self, run_path):
    self.language, self.model_name, self.run_name, self.total_iterations = self.read_model_info(run_path)

  def find_model_info_dir(self, model_dir: str) -> str:
      model_dir_name = os.path.basename(model_dir)
      if 'checkpoint' in model_dir_name:
          model_info_dir = os.path.dirname(model_dir)
      else:
          model_info_dir = model_dir
      return model_info_dir

  def find_model_info(self, model_dir: str) -> str:
      model_info_dir = self.find_model_info_dir(model_dir)
      model_info_path = model_info_dir + '/model_info.yml'
      assert os.path.exists(model_info_path)
      return model_info_path

  def read_model_info(self, model_dir: str):
      model_info_path = self.find_model_info(model_dir)
      with open(model_info_path) as f:
          fileContent = f.read()
          model_info = yaml.load(fileContent)
      return model_info['language'], model_info['model_name'], model_info['run_name'], model_info['total_iterations']

def get_subdirs(path: str) -> List[str]:
    walk = os.walk(path)
    _, dirs, _ = next(walk)
    return dirs
