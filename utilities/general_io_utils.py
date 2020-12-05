"""
general utilities for re-use
"""
from configparser import ConfigParser
import os
import pandas as pd
import pickle
from timelogging.timeLog import log
from typing import List, Tuple, Union, Optional

config_parser = ConfigParser()


def assertDirExistent(path):
    if not os.path.exists(path):
        raise IOError(f'{path} does not exist')


def assertFileInxestent(file_path):
    """ assert if file is inexistent
    :param: filePath"""
    if os.path.isfile(file_path):
        raise FileExistsError(f'{file_path} already exists')


def read_data(
        in_path: str,
        text_name: Optional[str] = None,
        summary_name: Optional[str] = None,
        limit: Optional[int] = None
) -> List[Union[Tuple, str]]:
    """general function to call all types of import functions

    Args:
        in_path (str): file path to read from
        text_name (Optional[str], optional): name of the text file.
        Defaults to None.
        summary_name (Optional[str], optional): name of the summary file.
        Defaults to None.
        limit (Optional[int], optional): data limitation.
        Defaults to None.

    Returns:
        List[Union[Tuple, str]]: all textes as list
    """
    if text_name is None \
            and summary_name is None:
        return read_single_txt(in_path)
    else:
        if all(".txt" in item for item in [text_name, summary_name]):
            return read_txt(in_path, text_name, summary_name, limit)
        elif all(".csv" in item for item in [text_name, summary_name]):
            return read_csv(in_path, text_name, summary_name, limit)
        elif all(".pickle" in item for item in [text_name, summary_name]):
            return read_pickle(in_path, text_name, summary_name, limit)
        else:
            log(f"{text_name} or {summary_name} is not supported!")
            exit()


def read_csv(
        in_path: str,
        text_name: str,
        summary_name: str,
        limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    """read data from csv file

    Args:
        in_path (str): file path to read from
        text_name (Optional[str], optional): name of the text file.
        Defaults to None.
        summary_name (Optional[str], optional): name of the summary file.
        Defaults to None.
        limit (Optional[int], optional): data limitation.
        Defaults to None.

    Returns:
        List[Tuple[str, str]]: listed text and assigned summary
    """

    df = pd.read_csv(in_path, escapechar="\\")

    data = [(text, summary)
            for text, summary in zip(df[text_name], df[summary_name])]

    if limit:
        data = data[:limit]

    return data


def read_pickle(
        in_path: str,
        text_name: str,
        summary_name: str,
        limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    """read data from pickle file

    Args:
        in_path (str): file path to read from
        text_name (Optional[str], optional): name of the text file.
        Defaults to None.
        summary_name (Optional[str], optional): name of the summary file.
        Defaults to None.
        limit (Optional[int], optional): data limitation.
        Defaults to None.

    Returns:
        List[Tuple[str, str]]: listed text and assigned summary
    """

    text_path = os.path.join(in_path, text_name)
    summary_path = os.path.join(in_path, summary_name)

    texts = pickle.load(open(text_path, mode="rb"))
    summaries = pickle.load(open(summary_path, mode="rb"))

    assert len(texts) == len(summaries)

    data = [(text, summary) for text, summary in zip(texts, summaries)]

    if limit:
        data = data[:limit]

    return data


def read_txt(
    in_path: str,
    texts_name: str,
    summary_name: str,
    limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    """read data from txt file

    Args:
        in_path (str): file path to read from
        text_name (Optional[str], optional): name of the text file.
        Defaults to None.
        summary_name (Optional[str], optional): name of the summary file.
        Defaults to None.
        limit (Optional[int], optional): data limitation.
        Defaults to None.

    Returns:
        List[Tuple[str, str]]: listed text and assigned summary
    """

    text_path = os.path.join(in_path, texts_name)
    summary_path = os.path.join(in_path, summary_name)

    with open(text_path, mode="r", encoding="utf-8") as text_handle:
        texts = [text.rstrip('\n') for text in text_handle.readlines()]

    with open(summary_path, mode="r", encoding="utf-8") as summary_handle:
        summaries = [summary.rstrip('\n')
                     for summary in summary_handle.readlines()]

    assert len(texts) == len(summaries)

    data = [(text, summary) for text, summary in zip(texts, summaries)]

    if limit:
        data = data[:limit]

    return data


def read_single_txt(file_path: str) -> List[str]:
    """read text/lines from single txt file

    Args:
        file_path (str): path to file

    Raises:
        FileNotFoundError: path is empty

    Returns:
        List[str]: listed texts
    """
    if not os.path.isfile(file_path):  # check if correct files exist
        raise FileNotFoundError(f'{file_path} not found')

    with open(file_path, mode="r", encoding="utf-8") as file_handle:
        text = file_handle.readlines()
        lines = [line.rstrip('\n') for line in text]

    return lines


def read_config(config_path: str) -> Tuple[dict, dict]:
    """read the .ini file which provides
       the configurations for the different
       pipeline runs

    Args:
        config_path (str): path to config .ini file

    Raises:
        FileNotFoundError: path is empty

    Returns:
        Tuple[dict, dict]: returns parameters 
        for config sections
    """
    config_dict = dict()
    log("Read from config", config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    config_parser.read(config_path)
    for section in config_parser.sections():
        config_dict.update({section: dict()})

        for entry in config_parser[section]:
            config_dict[section].update({entry: config_parser[section][entry]})
    try:
        return config_dict['MODEL'], config_dict['TRAINING']
    except Exception:
        return config_dict['MODEL'], config_dict['EVALUATION']


def check_make_dir(dir_or_file: str, create_dir: bool = False) -> bool:
    """check if file exists
       check if directory exist
       -> make if not

    Args:
        dir_or_file (str): path or directory
        create_dir (bool, optional): If not exists -> create.
        Defaults to False.

    Returns:
        bool: exists or not
    """
    ending = dir_or_file.split("/")[-1]
    if "." in ending:
        if not os.path.isfile(dir_or_file):
            return False
        return True
    else:
        if not os.path.exists(dir_or_file):
            if create_dir:
                os.makedirs(dir_or_file)
                return False
        else:
            return True


def write_txt(file_path: str, texts: List[str]):
    """write a list of texts to txt file

    Args:
        file_path (str): define output file
        texts (List[str]): texts to write
    """
    assertFileInxestent(file_path)
    with open(file_path, mode="w", encoding="utf-8") as file_handle:
        for text in texts:
            file_handle.write(text.rstrip("\n") + "\n")


def write_table(
    dictionary: dict,
    output_dir: str,
    file_name: str,
    file_format: str = "csv"):
    """write to excel or csv file given a dict

    Args:
        dictionary (dict): information to write
        output_dir (str): output diretory
        file_name (str): file name with ending
        file_format (str, optional): switch for excel or csv.
        Defaults to "csv".
    """
    
    assert file_format in [
        "csv", "excel"], "'file_format' has to be either 'excel' or 'csv'!"
    output_path = os.path.join(output_dir, file_name)
    log("\nWrite results to", output_path)
    df = pd.DataFrame.from_dict(dictionary, orient="columns")
    if file_format == "csv":
        output_path += ".csv"
        df.to_csv(output_path, sep=";")
    else:
        with pd.ExcelWriter(output_path + ".xlsx") as writer:
            df.to_excel(writer, "Overview")


def write_pickle(obj: Union[object, list], file_name: str, file_path: str):
    """write python object to binary

    Args:
        obj (Union[object, list]): object to save
        file_name (str): name of the binary
        file_path (str): path for the binary
    """
    if not file_name.endswith('.pickle'):
        file_name += ".pickle"

    pickle_path = os.path.join(file_path, file_name)
    log("\nSave pickle to", pickle_path)
    with open(pickle_path, mode="wb") as file_handle:
        pickle.dump(obj, file_handle)
