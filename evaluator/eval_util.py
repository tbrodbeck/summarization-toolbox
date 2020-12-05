"""
General utilities used for the evaluation
"""
import os
from typing import Tuple
import yaml

class ModelInfoReader:
    """class to process run info
    """
    def __init__(self, run_path: str):
        """set general class parameters

        Args:
            run_path (str): specifies run to evaluate
        """
        self.language, self.model_name, self.run_name \
            = self.read_model_info(run_path)

    @staticmethod
    def find_model_info_dir(model_dir: str) -> str:
        """search for the model in directory

        Args:
            model_dir (str): directory to check

        Returns:
            str: returns directory that contains info
        """
        model_dir_name = os.path.basename(model_dir)
        if 'checkpoint' in model_dir_name:
            model_info_dir = os.path.dirname(model_dir)
        else:
            model_info_dir = model_dir

        return model_info_dir

    def find_model_info(self, model_dir: str) -> str:
        """search for model info file

        Args:
            model_dir (str): directory that contains info

        Returns:
            str: path to info file
        """
        model_info_dir = self.find_model_info_dir(model_dir)
        model_info_path = model_info_dir + '/model_info.yml'

        assert os.path.exists(model_info_path)

        return model_info_path

    def read_model_info(self, model_dir: str) -> Tuple[str, str, str]:
        """read from info file

        Args:
            model_dir (str): directory with model

        Returns:
            Tuple[str, str, str]: language, model name and run name
        """
        model_info_path = self.find_model_info(model_dir)
        with open(model_info_path) as file_handle:
            file_content = file_handle.read()
            model_info = yaml.load(file_content)
        return model_info['language'], model_info['model_name'], model_info['run_name']
