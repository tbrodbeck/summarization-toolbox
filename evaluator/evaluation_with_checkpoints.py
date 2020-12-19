"""
main file to start the evaluation from
"""
import sys
sys.path.append(".")
import os
from typing import Dict, Optional
import torch
from timelogging.timeLog import log
from evaluator import eval_util
from evaluator.eval import Evaluator
from evaluator import metrics
from utilities.io_utils import check_make_dir
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from utilities.cleaning_utils import limit_data

DATA_DIR = "./dataProvider/datasets"
METRICS = ["SemanticSimilarity", "Rouge"]

def get_metric(metric_type: str, language: str) -> metrics.Metric:
    metric = None
    assert metric_type in METRICS, \
        f"{metric_type} is not a supported metric! \
            Please use one of those: {str(METRICS)}"
    if metric_type == "SemanticSimilarity":
        metric = metrics.SemanticSimilarityMetric(language)
    if metric_type == "Rouge":
        metric = metrics.Rouge()

    return metric

def get_checkpoint_iterations(checkpoint_dir: str) -> str:
    return checkpoint_dir.split("-")[1]

def initialize_model(model_dir: str, language: str) -> AbstractiveSummarizer:
    return AbstractiveSummarizer(
        model_dir,
        language,
        status="fine-tuned"
    )

def initialize_reference_model(language: str) -> AbstractiveSummarizer:
    return AbstractiveSummarizer(
        None,
        language,
        "base"
    )

def preprocess_data(dataset_name: str, nr_samples: int, tokenizer_name) -> Dict:
    data_set_dir = os.path.join(DATA_DIR, dataset_name)
    assert check_make_dir(data_set_dir), f"Data set '{dataset_name}' \
        not directory '{DATA_DIR}'. \
            Please store data there!"

    tensor_dir = os.path.join(
        data_set_dir,
        tokenizer_name
    )
    try:
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir)
    except Exception:
        tensor_dir += "_filtered"
        assert (check_make_dir(tensor_dir) and os.listdir(tensor_dir)), f"Neither '{tensor_dir.rstrip('_filtered')} not '{tensor_dir}' does exist or it is empty!"

    source_path = os.path.join(tensor_dir, "val_source.pt")
    target_path = os.path.join(tensor_dir, "val_target.pt")
    assert os.path.isfile(source_path) and os.path.isfile(target_path), f"Data pair '{source_path}' and '{target_path}' does not exist!"

    data_dict = {
        "source": torch.load(open(source_path, "rb")),
        "target": torch.load(open(target_path, "rb"))
    }
    return limit_data(data_dict, nr_samples)

def prepare_evaluator(tokenizer: object, metric_name: str, language: str, dataset_name: str, nr_samples: int, model_path: str, model_name: str) -> Evaluator:
    metric = get_metric(metric_name, language)
    data_dict = preprocess_data(dataset_name, nr_samples, model_name)
    return Evaluator(data_dict, metric, tokenizer)

def evaluate_with_checkpoints(run_path: str, dataset_name: str, nr_samples=10, metric_type='Rouge'):
    """ Considers all checkpoints and final model for evaluation generation

    Args:
        runPath (str): Path of run. E.g. `modelTrainer/results/t5-de/0`
        dataset_name (str): Name of dataset. E.g. `golem`
        nr_samples (int): Number of samples to evaluate on
    """
    model_info = eval_util.ModelInfoReader(run_path)
    evaluation_basepath = f'evaluator/evaluations/{model_info.run_name}'
    checkpoint_dirs = eval_util.get_subdirs(run_path)
    model = initialize_model(run_path, model_info.language)
    reference_model = initialize_reference_model(model_info.language)  # TODO in evaluator packen
    evaluator = prepare_evaluator(model.tokenizer, metric_type, model_info.language, dataset_name, nr_samples, run_path, model_info.model_name)
    # for checkpoint_dir in checkpoint_dirs:
    #     checkpoint_model = initialize_model(checkpoint_dir, model_info.language)
    #     log(f'Evaluating {checkpoint_dir}...')
    #     checkpoint_model_path = os.path.join(run_path, checkpoint_dir)
    #     iterations = get_checkpoint_iterations(checkpoint_dir)
    #     evaluate(checkpoint_model_path, dataset_name, model_info.language, model_info.model_name, output_dir=f"{evaluation_basepath}/{iterations}-iterations", number_samples=nr_samples)
    log(f'Evaluating final model...')

    evalutation_dict = evaluator.get_score_dict(model, reference_model)

    info_data_frame = evaluator.create_data_frame(evalutation_dict)

    evaluator.save_data_frame(info_data_frame, evaluation_basepath + "/Overview.xlsx", file_format="excel")

    # evaluate(run_path, dataset_name, model_info.language, model_info.model_name, output_dir=f"{evaluation_basepath}/{model_info.total_iterations}-iterations", number_samples=nr_samples)
    return evaluation_basepath


if __name__ == "__main__":
    evaluate(
        "./modelTrainer/results/t5-de/0/checkpoint-5000",
        "golem",
        "german",
        "WikinewsSum/t5-base-multi-de-wiki-news"
    )
