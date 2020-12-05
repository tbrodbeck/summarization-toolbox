"""
main file to start the evaluation from
"""
import sys
sys.path.append(".")
import os
from typing import Optional
import torch
from timelogging.timeLog import log
from evaluator import eval_util
from evaluator.eval import run_evaluation
from evaluator.metrics import SemanticSimilarityMetric
from utilities.general_io_utils import check_make_dir
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer

DATA_DIR = "./dataProvider/datasets"
METRICS = ["SemanticSimilarity"]


def evaluate(
        model_dir: str,
        data_set_name: str,
        language: str,
        model_name: str,
        output_dir: Optional[str] = "evaluator/output",
        number_samples: Optional[int] = 5,
        reference_model: Optional[bool] = False,
        metric: Optional[str] = "SemanticSimilarity"):
    """set infrastructure for the evaluation and run it afterwards

    Args:
        model_dir (str): directory to load model from
        data_set_name (str): name of data to evaluate on
        language (str): choose between 'english' or 'german'
        model_name (str): name of the model
        (needed for base model and data folder structure)
        output_dir (Optional[str], optional): directory to store evaluation file.
        Defaults to "evaluator/output".
        number_samples (Optional[int], optional): Number of samples to evluate on.
        Defaults to 5.
        reference_model (Optional[bool], optional): model to use as a baseline.
         Defaults to False.
        metric (Optional[str], optional): Choose the name of a metric used for evaluation.
        Defaults to "SemanticSimilarity".
    """

    evaluation_parameters = {
        "language": language,
        "output_directory": output_dir,
        "number_samples": number_samples,
        "reference_model": reference_model,
        "metric": metric
    }
    print("\n")
    log("Received parameters for evaluation:")
    for param in evaluation_parameters:
        log(f"- {param}: {evaluation_parameters[param]}")

    ###################################
    # Initialize model
    ###################################
    model = AbstractiveSummarizer(
        model_dir,
        evaluation_parameters["language"],
        status="fine-tuned"
    )

    # initialize reference model
    if evaluation_parameters["reference_model"] == "True":
        reference_model = AbstractiveSummarizer(
            evaluation_parameters["language"],
            "base"
        )
    else:
        reference_model = None

    # check if output directory exists
    out_dir = evaluation_parameters['output_directory']
    if not check_make_dir(out_dir, create_dir=True):
        log(f"Created output directory'{out_dir}'")

    samples = int(evaluation_parameters['number_samples'])

    ###################################
    # Load evaluation data
    ###################################
    data_set_dir = os.path.join(DATA_DIR, data_set_name)
    assert check_make_dir(data_set_dir), f"Data set '{data_set_name}' \
        not directory '{DATA_DIR}'. \
            Please store data there!"

    tensor_dir = os.path.join(
        data_set_dir,
        model_name
    )
    try:
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir)
    except FileNotFoundError:
        tensor_dir += "_filtered"
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir), \
            f"Neither '{tensor_dir.rstrip('_filtered')}' \
                not '{tensor_dir}' does exist or it is empty!"

    source_path = os.path.join(tensor_dir, "val_source.pt")
    target_path = os.path.join(tensor_dir, "val_target.pt")
    assert os.path.isfile(source_path) and os.path.isfile(target_path), \
        f"Data pair '{source_path}' and '{target_path}' does not exist!"

    evaluation_dict = {
        "source": torch.load(open(source_path, "rb")),
        "target": torch.load(open(target_path, "rb"))
    }

    metric = None
    assert evaluation_parameters["metric"] in METRICS, \
        f"{evaluation_parameters['metric']} is not a supported metric! \
            Please use one of those: {str(METRICS)}"
    if evaluation_parameters["metric"] == "SemanticSimilarity":
        metric = SemanticSimilarityMetric(evaluation_parameters["language"])

    run_evaluation(evaluation_dict, model, metric,
                   out_dir, samples, reference_model)


def get_checkpoint_iterations(checkpoint_dir: str) -> str:
    return checkpoint_dir.split("-")[1]

def evaluate_with_checkpoints(run_path: str, dataset_name: str, nr_samples: int = 10):
    """ Considers all checkpoints and final model for evaluation generation

    Args:
        runPath (str): Path of run. E.g. `modelTrainer/results/t5-de/0`
        dataset_name (str): Name of dataset. E.g. `golem`
        nr_samples (int): Number of samples to evaluate on
    """
    model_info = eval_util.ModelInfoReader(run_path)
    evaluation_basepath = f'evaluator/evaluations/{model_info.run_name}'
    checkpoint_dirs = eval_util.get_subdirs(run_path)
    for checkpoint_dir in checkpoint_dirs:
        log(f'Evaluating {checkpoint_dir}...')
        checkpoint_model_path = os.path.join(run_path, checkpoint_dir)
        iterations = get_checkpoint_iterations(checkpoint_dir)
        evaluate(checkpoint_model_path, dataset_name, model_info.language, model_info.model_name, output_dir=f"{evaluation_basepath}/{iterations}-iterations", number_samples=nr_samples)
    log(f'Evaluating final model...')
    evaluate(run_path, dataset_name, model_info.language, model_info.model_name, output_dir=f"{evaluation_basepath}/{model_info.total_iterations}-iterations", number_samples=nr_samples)
    return evaluation_basepath
