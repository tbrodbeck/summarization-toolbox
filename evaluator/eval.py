"""
orchestrate the evaluation
"""
from typing import Union, Optional
import torch
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from evaluator.metrics import Metric
from utilities.gerneral_io_utils import write_table
from utilities.cleaning_utils import limit_data


def run_evaluation(
        data: dict,
        model: AbstractiveSummarizer,
        metric: Metric,
        out_dir: str,
        samples: int,
        base_model: Optional[AbstractiveSummarizer] = None):
    """orchestrator for the evaluation process

    Args:
        data (dict): evaluation data dictionary
        model (AbstractiveSummarizer): model to be evaluated
        metric (Metric): choosen metric for the evaluation
        out_dir (str): directory to store results like evaluation excel/csv
        samples (int): number of texts to evaluate on
        base_model (Optional[AbstractiveSummarizer], optional): 
        Not fine-tuned model as baseline. Defaults to None.
    """
    # limit data to evaluate
    data = limit_data(data, samples)
    # calculate different scores
    calculate_scores(data, metric, model, base_model, out_dir)


def calculate_scores(
        data: dict,
        metric: Metric,
        model: AbstractiveSummarizer,
        base_model: Union[None, AbstractiveSummarizer],
        out_dir: str):
    """calculate the matric scores for the fine tuned model
       given the validation set

    Args:
        data (dict): validation set
        metric (Metric): supported metric
        model (AbstractiveSummarizer): fine tuned model
        base_model (Union[None, AbstractiveSummarizer]): baseline model
        out_dir (str): store output
    """
    # calculate predictions
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    source_texts = model.tokenizer.batch_decode(
        data['source']['input_ids'].to(device))
    predicted_summaries = model.predict(data['source'])
    target_summaries = model.tokenizer.batch_decode(
        data['target']['input_ids'].to(device))

    # only if base model given
    if base_model:
        base_predictions = base_model.predict(data['source'])
        base_scores = list()

    gold_scores = list()
    fine_tuned_scores = list()
    similarity_scores = list()

    # calculate metrics
    for i, (source_text, predicted_summary, target_summary) in \
            enumerate(zip(source_texts, predicted_summaries, target_summaries)):

        gold_prediction_score = metric.get_score(target_summary, source_text)
        fine_tuned_prediction_score = metric.get_score(
            predicted_summary, source_text)
        summary_similarity_score = metric.get_score(
            predicted_summary, target_summary)

        gold_scores.append(gold_prediction_score)
        fine_tuned_scores.append(fine_tuned_prediction_score)
        similarity_scores.append(summary_similarity_score)

        if base_model:
            base_prediction_score = metric.get_score(
                base_predictions[i], source_text)
            base_scores.append(base_prediction_score)

    # create dict to prepare
    # data for excel/csv output
    overview_dict = {
        "text": source_texts,
        "target_summary": target_summaries,
        "fine_tuned_summary": predicted_summaries
    }
    if base_model:
        overview_dict["base_model_summary"] = base_predictions

    overview_dict.update(
        {
            "gold_score": gold_scores,
            "fine_tuned_score": fine_tuned_scores,
            "summary_similarity_score": similarity_scores
        }
    )

    if base_model:
        overview_dict["base_model_score"] = base_scores

    # produce overview as excel or csv
    write_table(overview_dict, out_dir, "Overview", "excel")
