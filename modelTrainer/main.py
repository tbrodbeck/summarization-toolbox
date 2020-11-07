"""
main file wich runs the fine tuning pipeline:

CLI:
to provide pipeline parameters via CLI:
python modelTrainer.mai.py -d DATASETNAME -m MODELNAME -c CONFIGNAME
"""
import sys
sys.path.append(".")
from fire import Fire
import os
from timelogging.timeLog import log
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from modelTrainer.fine_tuning import fine_tune_model
from utilities.gerneral_io_utils import read_config, check_make_dir
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

# required parameters
# for model
MODEL_CONFIG = [
    "language",
    "version",
    "status",
    "freezed_components",
    "output_directory"
]

# required parameters
# for training
TRAINING_CONFIG = [
    "epochs",
    "train_batch_size",
    "val_batch_size"
]

DATA_DIRECTORY = "./dataProvider/datasets/"

def initialize_trainer(dataset_name: str, model_name: str, config_name: str = "fine_tuning_config.ini"):
    """
    set up for training process
    :param dataset_name:
    :param model_name:
    :param config_name:
    :return:
    """
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

    # check data name available
    assert dataset_name in dataset_names, \
        f"'{dataset_name}' not in available datasets: {dataset_names}"

    # check tensors folder
    dataset_dir = os.path.join(DATA_DIRECTORY, dataset_name)
    assert check_make_dir(dataset_dir + "/tensors"), \
        f"No 'tensors' folder in '{dataset_dir}'"

    tensor_dir = os.path.join(dataset_dir, "tensors")
    data_files = [file for file in os.listdir(tensor_dir)
                  if '.pt' in file]

    # check training files
    assert len(data_files) > 0, \
        f"'{tensor_dir}' is empty! Please provide '.pt' files!"

    data_dict = dict()
    for split_name in SPLIT_NAMES:
        if "test" not in split_name:
            files = list()
            for text_name in TEXT_NAMES:
                files.append(f"{split_name}_{text_name}.pt")

            if files:
                # check data file pairs
                assert all([check_make_dir(tensor_dir + "/" + file) for file in files]), \
                    f"'{files[0]}'/'{files[0]}' pair doesn't exist in '{dataset_dir}'!"

                data_dict[split_name] = dict()
                for text_name in TEXT_NAMES:
                    file_path = os.path.join(
                        tensor_dir,
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
    MODEL, TRAINING = read_config(
        config_path
    )

    # model parameters
    model_parameters = dict()
    for parameter_name in MODEL_CONFIG:
        if MODEL[parameter_name]:
            model_parameters[parameter_name] = MODEL[parameter_name]

    print("\n")
    log("\nReceived parameters for model:")
    for p in model_parameters:
        log(f"- {p}: {model_parameters[p]}")
    print("\n")

    # check if output directory exists
    if not check_make_dir(model_parameters["output_directory"], create_dir=True):
        log(f"Created output directory'{model_parameters['output_directory']}'")

    ###################################
    # Initialize Model
    ###################################

    # initialize summary model
    model = AbstractiveSummarizer(
        model_parameters["language"],
        model_parameters["status"],
        model_parameters["output_directory"],
        int(model_parameters["version"]),
        model_parameters["freezed_components"].split(";")
    )

    ###################################
    # Run fine tuning
    ###################################

    # training parameters
    training_parameters = dict()
    for parameter_name in TRAINING_CONFIG:
        if TRAINING[parameter_name]:
            training_parameters[parameter_name] = TRAINING[parameter_name]

    print("\n")
    log("Received parameters for training:")
    for p in training_parameters:
        log(f"- {p}: {training_parameters[p]}")

    log("\n+++ FINE-TUNING +++")

    fine_tune_model(
        model,
        model_parameters["output_directory"],
        data_dict,
        training_parameters
    )


if __name__ == '__main__':
    Fire(initialize_trainer)
