from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from utilities.gerneral_io_utils import write_excel
from tqdm import tqdm

def run_evaluation(data: dict, model: AbstractiveSummarizer, metric, out_dir, samples, base_model: AbstractiveSummarizer = None):
    data = limit_data(data, samples)
    calculate_scores(data, metric, model, base_model, out_dir)

def calculate_scores(data, metric, model, base_model, out_dir):

    predicted_summary = model.predict(data['source'])
    print(predicted_summary)


def limit_data(data_dict: dict, limit: int = -1):
    """
    limit the evaluation samples
    :param data_dict:
    :param limit:
    :return:
    """
    if limit == -1:
        return data_dict

    new_dict = dict()
    for item in ["source", "target"]:
        new_dict[item] = {
            "input_ids": data_dict[item]['input_ids'][:limit],
            "attention_mask": data_dict[item]['attention_mask'][:limit]
        }
    return new_dict


def produce_overview(model, texts, targets, metric, output_path):
    """
    create csv with evaluation
    :param metric:
    :param output_path:
    :param model:
    :param texts:
    :param targets:
    :return:
    """
    all_predictions = list()
    all_similarities = list()
    all_bases = list()
    all_summaries = list()

    for text, target in tqdm(zip(texts, targets)):
        prediction = model.predict(text)
        similarity_score = metric.get_score(prediction, target)
        base_score = metric.get_score(text, target)
        summary_score = metric.get_score(text, prediction)

        all_predictions.append(prediction)
        all_similarities.append(similarity_score)
        all_bases.append(base_score)
        all_summaries.append(summary_score)

    overview_dict = {
        "text": texts,
        "target": targets,
        "prediction": all_predictions,
        "similarity score": all_similarities,
        "summary score": all_summaries,
        "base score": all_bases
    }

    write_excel(overview_dict, output_path, "evaluation.xlsx")