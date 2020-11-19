import sys
sys.path.append(".")

import fire
import os

from timelogging.timeLog import log
import torch
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from utilities.gerneral_io_utils import check_make_dir, read_config
from evaluator.metrics import SemanticSimilarityMetric
from evaluator.eval import run_evaluation


DATA_DIR = "./dataProvider/datasets"

MODEL_NAME = [
    "t5-base",
    "WikinewsSum/t5-base-multi-de-wiki-news"
]

# required parameters
# for model
MODEL_CONFIG = [
    "language",
    "version",
    "status",
    "model_directory",
    "checkpoint"
]

# required parameters
# for evaluation
EVALUTATION_CONFIG = [
    "metric",
    "output_directory",
    "samples"
]

def evaluate(data_set_name: str, model_name: str, config_path: str = "./evaluator/config/evaluation_config.ini", reference_to_base: bool = False):
    """
    run evaluation
    :param reference_to_base:
    :param data_set_name:
    :param model_name:
    :param config_path:
    :return:
    """

    ###################################
    # Read from config
    ###################################
    MODEL, EVALUATION = read_config(
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

    evaluation_parameters = dict()
    for parameter_name in EVALUTATION_CONFIG:
        if EVALUATION[parameter_name]:
            evaluation_parameters[parameter_name] = EVALUATION[parameter_name]

    print("\n")
    log("Received parameters for evaluation:")
    for p in evaluation_parameters:
        log(f"- {p}: {evaluation_parameters[p]}")

    ###################################
    # Initialize model
    ###################################
    model = AbstractiveSummarizer(
        model_parameters["language"],
        model_parameters["status"],
        model_parameters["model_directory"],
        int(model_parameters["version"]),
        checkpoint=None if model_parameters["checkpoint"] == "None" \
            else model_parameters["checkpoint"]
    )

    # initialize reference model

    if reference_to_base:
        reference_model = AbstractiveSummarizer(
            model_parameters["language"],
            "base"
        )
    else:
        reference_model = None

    # check if output directory exists
    out_dir = evaluation_parameters['output_directory']
    if not check_make_dir(out_dir, create_dir=True):
        log(f"Created output directory'{out_dir}'")

    samples = int(evaluation_parameters['samples'])

    ###################################
    # Load evaluation data
    ###################################
    data_set_dir = os.path.join(DATA_DIR, data_set_name)
    assert check_make_dir(data_set_dir), f"Data set '{data_set_name}' not directory '{DATA_DIR}'. " \
                                         f"Please store data there!"

    model_folder = model_name.split("/")[0]
    tensor_folder = model_name.split("/")[1]
    tensor_dir = os.path.join(
        data_set_dir,
        model_folder,
        tensor_folder
    )
    try:
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir)
    except:
        tensor_dir += "_filtered"
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir), \
            f"Neither '{tensor_dir.rstrip('_filtered')}' not '{tensor_dir}' does exist or it is empty!"

    source_path = os.path.join(tensor_dir, "test_source.pt")
    target_path = os.path.join(tensor_dir, "test_target.pt")
    assert os.path.isfile(source_path) and os.path.isfile(target_path), \
        f"Data pair '{source_path}' and '{target_path}' does not exist!"

    evaluation_dict = {
        "source": torch.load(open(source_path, "rb")),
        "target": torch.load(open(target_path, "rb"))
    }

    metric = None
    if evaluation_parameters["metric"] == "SemanticSimilarity":
        metric = SemanticSimilarityMetric(model_parameters["language"])

    run_evaluation(evaluation_dict, model, metric, out_dir, samples, reference_model)


if __name__ == '__main__':
  fire.Fire(evaluate)
