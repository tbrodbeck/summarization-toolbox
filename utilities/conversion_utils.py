"""
general utils to convert one
file format into another
"""

import pickle
from typing import Union

def convert_txt_to_pickle(in_paths: Union[str, list], out_paths: Union[str, list]):
    """
    txt to pickle
    :param out_paths:
    :param in_paths:
    :return:
    """

    if isinstance(in_paths, list):
        assert isinstance(out_paths, list)
        assert len(out_paths) == len(in_paths)
    else:
        in_paths = [in_paths]
        out_paths = [out_paths]

    for i, path in enumerate(in_paths):
        print(f"Reading {path}")
        with open(path, mode="r", encoding="utf-8") as file_handle:
            lines = file_handle.readlines()
            all_lines = [line.rstrip("\n") for line in lines]

        with open(out_paths[i], mode="wb") as pickle_handle:
            print("Save as pickle.")
            pickle.dump(all_lines, pickle_handle)