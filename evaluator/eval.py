"""
orchestrate the evaluation
"""
from typing import Union, Optional, List, Tuple
import pandas as pd
import torch
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from evaluator.metrics import Metric
from utilities.general_io_utils import write_table
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
    #calculate_scores(data, metric, model, base_model, out_dir)
    evaluator = Evaluator(data, metric, model, base_model)

    evaluator.update_predictions(update_base=bool(base_model))

    source_embeddings = evaluator.get_sentence_embeddings(
        evaluator.source_text
    )
    target_embeddings = evaluator.get_sentence_embeddings(
        evaluator.target_text
    )
    prediction_embeddings = evaluator.get_sentence_embeddings(
        evaluator.predictions
    )
    summary_dict = {
        "source": evaluator.source_text,
        "target": evaluator.target_text,
        "prediction": evaluator.predictions
    }
    score_dict = {
        "gold_score": evaluator.get_similarities(source_embeddings, target_embeddings),
        "prediction_score": evaluator.get_similarities(source_embeddings, prediction_embeddings),
        "summary_similarity_score": evaluator.get_similarities(target_embeddings, prediction_embeddings)
    }

    if base_model:
        base_prediction_embeddings = evaluator.get_sentence_embeddings(evaluator.base_predictions)
        summary_dict.update(
            {"base_prediction": evaluator.base_predictions}
        )
        score_dict.update(
            {
                "base_prediction_score": evaluator.get_similarities(source_embeddings, base_prediction_embeddings),
                "base_summary_similarity_score": evaluator.get_similarities(target_embeddings, prediction_embeddings)
            })

    info_data_frame = evaluator.create_data_frame({**summary_dict,**score_dict})

    evaluator.save_data_frame(info_data_frame, out_dir + "/Overview.xlsx", "excel")

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


class Evaluator:
    """general class to perform efficient evaluation
    """
    def __init__(
        self,
        data: dict,
        metric: Metric,
        model: AbstractiveSummarizer,
        base_model: Optional[AbstractiveSummarizer] = None):
        """set evaluation framework

        Args:
            data (dict): evaluation data dictionary
            model (AbstractiveSummarizer): model to be evaluated
            metric (Metric): choosen metric for the evaluation
            base_model (Optional[AbstractiveSummarizer], optional): 
            Not fine-tuned model as baseline. Defaults to None.
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        self.data = data
        self.model = model
        self.metric = metric
        self.base_model = base_model

        self.source_text, self.target_text = self.decode_tokens(data)
        self.predictions = None
        self.base_predictions = None


    def decode_tokens(self, data: dict):
        """turn tokens to text

        Args:
            data (dict): source and target texts

        Returns:
            [type]: [description]
        """
        source_tokens = self.model.tokenizer.batch_decode(
            data['source']['input_ids'].to(self.device)
        )
        target_tokens = self.model.tokenizer.batch_decode(
            data['target']['input_ids'].to(self.device)
        )
        return source_tokens, target_tokens
    
    def get_sentence_embeddings(self, text: Union[str, list]) -> list:

        if isinstance(text, str):
            text = [text]

        all_embeddings = list()
        for item in text:
            all_embeddings.append(
                self.metric.get_embedding(item)
            )

        return all_embeddings

    def get_similarities(
        self, 
        embedding_1: Union[list, torch.Tensor],
        embedding_2: Union[list, torch.Tensor]
        ) -> list:

        if isinstance(embedding_1, torch.Tensor):
            embedding_1 = [embedding_1]
            embedding_2 = [embedding_2]
        
        all_scores = list()
        for vector_1, vector_2 in zip(embedding_1, embedding_2):
            all_scores.append(
                self.metric.get_similarity(vector_1, vector_2)
            )
        return all_scores
    
    def update_model(self, new_model: AbstractiveSummarizer):
        self.model = new_model
    
    def update_predictions(self, update_base: bool):
        self.predictions = self.model.predict(self.data["source"])
        if update_base:
            self.base_predictions = self.base_model.predict(self.data["source"])
    
    @staticmethod
    def create_data_frame(info_dict: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(info_dict, orient="columns")
    
    @staticmethod
    def save_data_frame(
        data_frame: pd.DataFrame,
        output_path: str,
        file_format: Optional[str] = "csv"
    ):
        if file_format == "csv":
            output_path += ".csv"
            data_frame.to_csv(output_path, sep=";")
        else:
            with pd.ExcelWriter(output_path + ".xlsx") as writer:
                data_frame.to_excel(writer, "Overview")

