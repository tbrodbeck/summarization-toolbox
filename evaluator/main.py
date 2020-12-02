import sys
sys.path.append(".")
import os
import torch
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from utilities.gerneral_io_utils import check_make_dir, read_single_txt
from evaluator.metrics import SemanticSimilarityMetric
from evaluator.eval import run_evaluation
from evaluator import eval_util
from timelogging.timeLog import log

DATA_DIR = "./dataProvider/datasets"

def evaluate(model_dir: str, data_set_name: str, language: str, model_name: str, output_dir="evaluator/output", number_samples=5, reference_model=True, metric="SemanticSimilarity"):
    """
    run evaluation
    :param reference_to_base:
    :param data_set_name:
    :param model_name:
    :param config_path:
    :return:
    """

    evaluation_parameters = {
        "language": language,
        "output_directory": output_dir,
        "number_samples": number_samples,
        "reference_model": False,
        "metric": metric
    }
    print("\n")
    log("Received parameters for evaluation:")
    for p in evaluation_parameters:
        log(f"- {p}: {evaluation_parameters[p]}")

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
    assert check_make_dir(data_set_dir), f"Data set '{data_set_name}' not directory '{DATA_DIR}'. " \
                                         f"Please store data there!"


    tensor_dir = os.path.join(
        data_set_dir,
        model_name
    )
    try:
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir)
    except:
        tensor_dir += "_filtered"
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir), \
            f"Neither '{tensor_dir.rstrip('_filtered')}' not '{tensor_dir}' does exist or it is empty!"

    source_path = os.path.join(tensor_dir, "val_source.pt")
    target_path = os.path.join(tensor_dir, "val_target.pt")
    assert os.path.isfile(source_path) and os.path.isfile(target_path), \
        f"Data pair '{source_path}' and '{target_path}' does not exist!"

    evaluation_dict = {
        "source": torch.load(open(source_path, "rb")),
        "target": torch.load(open(target_path, "rb"))
    }

    metric = None
    if evaluation_parameters["metric"] == "SemanticSimilarity":
        metric = SemanticSimilarityMetric(evaluation_parameters["language"])

    run_evaluation(evaluation_dict, model, metric, out_dir, samples, reference_model)

def evaluate_with_checkpoints(run_path: str, dataset_name: str, nr_samples: int = 10):
    """ Considers all checkpoints and final model for evaluation generation

    Args:
        runPath (str): Path of run. E.g. `modelTrainer/results/t5-de/0`
        dataset_name (str): Name of dataset. E.g. `golem`
        nr_samples (int): Number of samples to evaluate on
    """
    model_info = eval_util.ModelInfoReader(run_path)
    evaluation_basepath = f'evaluator/evaluations/{model_info.run_name}'
    run_path_walk = os.walk(run_path)
    _, checkpoint_dirs, _ = next(run_path_walk)
    for checkpoint_dir in checkpoint_dirs:
        log(f'Evaluating {checkpoint_dir}...')
        checkpoint_model_path = os.path.join(run_path, checkpoint_dir)
        evaluate(checkpoint_model_path, dataset_name, model_info.language, model_info.model_name, output_dir=f"{evaluation_basepath}/{checkpoint_dir}", number_samples=nr_samples)
    log(f'Evaluating final model...')
    evaluate(run_path, dataset_name, model_info.language, model_info.model_name, output_dir=evaluation_basepath, number_samples=nr_samples)
