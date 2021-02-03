"""
main file wich runs the fine tuning pipeline:

CLI:
to provide pipeline parameters via CLI:
python modelTrainer.mai.py DATASETNAME MODELNAME
"""
import os
from typing import Optional
import sys
sys.path.append(".")
import torch
import fire

from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from modelTrainer.fine_tuning import fine_tune_model
from utilities.io_utils import read_config, check_make_dir


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
    "model_directory",
    "output_directory",
    "checkpoint"
]

# required parameters
# for training
TRAINING_CONFIG = [
    "epochs",
    "train_batch_size",
    "val_batch_size",
    "checkpoint_steps",
    "number_samples",
    "limit_val_data",
    "logging_steps",
    "eval_steps",
    "weight_decay",
    "warmup_steps"
]

DATA_DIRECTORY = "./dataProvider/datasets/"


def initialize_trainer(
        dataset_name: str,
        model_name: str,
        filtered: bool = True,
        config_name: Optional[str] = "fine_tuning_config.ini"):
    """fine tuning pipeline initialization

    Args:
        dataset_name (str): name of the dataset used for training
        model_name (str): model to fine tune on
        filtered (bool, optional): choose filtered or unfiltered tensors for training. Defaults to True.
        config_name (Optional[str], optional): name of config file.
        Defaults to "fine_tuning_config.ini".
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
    if filtered:
        filter_str = "_filtered"
    else:
        filter_str = ""
    try:
        tensor_dir = os.path.join(dataset_dir, model_name + filter_str)
        assert os.path.isdir(tensor_dir)
    except:
        raise AssertionError(f"Neither '{tensor_dir} nor '{tensor_dir}_filtered' exists!")
        
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

            if all([check_make_dir(tensor_dir + "/" + file) for file in files]):
                data_dict[split_name] = dict()
                for text_name in TEXT_NAMES:
                    file_path = os.path.join(
                        tensor_dir,
                        f"{split_name}_{text_name}.pt"
                    )
                    data_dict[split_name][text_name] = torch.load(
                        open(file_path, "rb")
                    )

    # check model is supported
    assert model_name in MODEL_NAMES, \
        f"'{model_name}' not supported. Please choose one of {MODEL_NAMES}"

    # set to default config if not given
    if config_name is None:
        config_path = "fine_tuning_config.ini"
    else:
        # check .ini file
        assert ".ini" in config_name, \
            "Config has to be an '.ini' file!"
        config_path = os.path.join("./modelTrainer/config", config_name)

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


    # check if output directory exists
    check_make_dir(model_parameters["output_directory"], create_dir=True)

    ###################################
    # Initialize Model
    ###################################

    # initialize summary model
    model = AbstractiveSummarizer(
        model_parameters["model_directory"],
        model_parameters["language"],
        model_parameters["status"]
    )
    if model_parameters["freezed_components"] != "None":
        model.freeze_model_layers(model_parameters["freezed_components"].strip().split(";"))
    ###################################
    # Run fine tuning
    ###################################

    # training parameters
    training_parameters = dict()
    for parameter_name in TRAINING_CONFIG:
        if TRAINING[parameter_name]:
            training_parameters[parameter_name] = TRAINING[parameter_name]

    fine_tune_model(
        model,
        model_parameters["output_directory"],
        data_dict,
        training_parameters
    )


if __name__ == '__main__':
    fire.Fire(initialize_trainer)
