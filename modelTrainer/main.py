"""
main file wich runs the fine tuning pipeline:

CLI:
to provide pipeline parameters via CLI:
python modelTrainer.mai.py -d DATASETNAME -m MODELNAME -c CONFIGNAME
"""
import sys
sys.path.append(".")
import os
from timelogging.timeLog import log
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from modelTrainer.fine_tuning import fine_tune_model
from utilities.gerneral_io_utils import read_config, check_make_dir
from utilities.parser_utils import parser
import torch

CLI = [
    "dataset",
    "model",
    "config"
]


MODEL_NAMES = [
    'WikinewsSum/t5-base-multi-de-wiki-news'
]

SPLIT_NAMES = [
    "train",
    "val",
    "test"
]
TEXT_NAMES = [
    'source',
    'target'
]

MODEL_CONFIG = []

DATA_DIRECTORY = "./dataProvider/datasets/"

if __name__ == '__main__':
    ###################################
    # Perform checks
    ###################################

    # check data path exists
    assert check_make_dir(DATA_DIRECTORY), \
        f"Make sure directory {DATA_DIRECTORY} exists!"

    # check that there are data folders
    dataset_names = [folder for folder in os.listdir(DATA_DIRECTORY)
                     if os.path.isdir(DATA_DIRECTORY + folder)]

    # check dataset folders
    assert len(dataset_names) > 0, \
        f"Directory '{DATA_DIRECTORY}' is empty!"

    ###################################
    # Initialize Parameters
    ###################################

    # parser automatically checks
    # if no input is given
    args = parser(*CLI)
    dataset_name = args.dataset
    model_name = args.model
    config_name = args.config

    # check data name available
    assert dataset_name in dataset_names, \
        f"'{dataset_name}' not in available datasets: {dataset_names}"

    dataset_dir = os.path.join(DATA_DIRECTORY, dataset_name)
    data_files = [file for file in os.listdir(dataset_dir)
                  if '.pt' in file]

    # check training files
    assert len(data_files) > 0, \
        f"'{dataset_dir}' is empty! Please provide '.pt' files!"

    data_dict = dict()
    for split_name in SPLIT_NAMES:
        files = list()
        for text_name in TEXT_NAMES:
            files.append(f"{split_name}_{text_name}.pt")

        if files:
            # check data file pairs
            assert all([check_make_dir(dataset_dir + "/" + file) for file in files]), \
                f"'{files[0]}'/'{files[0]}' pair doesn't exist in '{dataset_dir}'!"

            data_dict[split_name] = dict()
            for text_name in TEXT_NAMES:
                file_path = os.path.join(
                    dataset_dir,
                    f"{split_name}_{text_name}.pt"
                )

                log(f"load data from: {file_path}")
                data_dict[split_name][text_name] = torch.load(
                    open(file_path, "rb")
                )


    # check model is supported
    assert model_name in MODEL_NAMES, \
        f"'{model_name}' not supported. Please choose one of {MODEL_NAMES}"


    # set to default config if not given
    if config_name is None:
        config_path = "./config/fine_tuning_config.ini"
    else:
        # check .ini file
        assert ".ini" in config_name, \
            "Config has to be an '.ini' file!"
        config_path = os.path.join("./config", config_name)

    # check config file exists
    assert check_make_dir(config_path), \
        f"'{config_path}' doesn't exist. Please provide config a file!"

    ###################################
    # Read from config
    ###################################
    MODEL = read_config(
        config_path
    )

    # model parameters
    parameters = dict()
    for parameter_name in MODEL_CONFIG:
        if MODEL[parameter_name]:
            parameters[parameter_name] = MODEL[parameter_name]

    log("Received parameters:")
    for p in parameters:
        log(f"{p}: {parameters[p]}")

    # check if output directory exists
    if not check_make_dir(parameters["output_directory"], create_dir=True):
        log(f"Created output directory'{parameters['output_directory']}'")

    ###################################
    # Initialize Model
    ###################################

    # initialize summary model
    model = AbstractiveSummarizer(
        parameters["language"],
        parameters["status"],
        parameters["output_path"],
        parameters["version"],
        parameters["freezed_components"]
    )

    ###################################
    # Run fine tuning
    ###################################

    log("\n+++ FINE-TUNING +++")

    fine_tune_model(
        model,
        parameters["input_path"],
        parameters["output_path"],
        parameters["corpus_file"],
        parameters["target_file"],
        int(parameters["batch_size"]),
        parameters["freezed_components"]
    )