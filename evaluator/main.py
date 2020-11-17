import sys
sys.path.append(".")
import fire
import os
import sentence_transformers
from timelogging.timeLog import log
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import tqdm
from typing import List
import utilities
from utilities.gerneral_io_utils import read_single_txt

class SemanticSimilarityMetric:
    """
    calculates cosine similarity
    of word embeddings
    """
    def __init__(self, language: str):

        assert language in ["en", "de"]

        self.transformer = sentence_transformers.SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        if language == "en":
            print("Intitialized semantic similarity metric for English texts.")
        else:
            print("Intitialized semantic similarity metric for German texts.")


    def get_score(self, prediction: str, target: str) -> float:
        """
        get the similarity score
        for a single text pair
        :param prediction:
        :param target:
        :return:
        """

        prediction_embeddings = torch.tensor(self.transformer.encode(prediction)).unsqueeze(0)
        target_embeddings = torch.tensor(self.transformer.encode(target)).unsqueeze(0)

        score = self.cosine_similarity(prediction_embeddings, target_embeddings)

        return score.item()


    def produce_overview(self, model, texts, targets, output_path):
        """
        create csv with evaluation
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

        for text, target in tqdm.tqdm(zip(texts, targets)):
            prediction = model.predict(text)
            similarity_score = self.get_score(prediction, target)
            base_score = self.get_score(text, target)
            summary_score = self.get_score(text, prediction)

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

        utilities.gerneral_io_utils.write_excel(overview_dict, output_path, "evaluation.xlsx")

def evaluate_model(model: AutoModelWithLMHead,
                   input_path: str,
                   text_name: str,
                   target_name: str,
                   output_path: str = None):
    """
    evaluate process

    :param target_name:
    :param text_name:
    :param input_path:
    :param model:
    :param output_path: if output is provided
                        it automatically produces
                        an excel sheet as overview
    :return:
    """

    # put model in evaluation mode
    semantic_similiarity = SemanticSimilarityMetric("de")
    model.set_mode("eval")

    data = utilities.gerneral_io_utils.read_data(input_path, text_name, target_name, limit=100)

    if output_path is None:
        for text, target in data:
            summary = model.predict(text)
            similarity_score = semantic_similiarity.get_score(summary, target)
            print("\nText:")
            print(text)
            print("\nTarget:")
            print(target)
            print("\nScore:", similarity_score)
    else:
        print("\nProducing overview...")
        texts = [item[0] for item in data]
        targets = [item[1] for item in data]
        semantic_similiarity.produce_overview(model, texts, targets, output_path)

def predict(modelPaths: List):
    for modelPath in modelPaths:
        model = AutoModelWithLMHead.from_pretrained(modelPath)

def evaluate(runPath: str, tokenizerName: str, modelName: str, datasetName: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
    maxTokenSize = tokenizer.max_model_input_sizes[modelName]
    dataDir = f'dataProvider/datasets/{datasetName}/'
    sourceText = read_single_txt('{}{}.{}'.format(dataDir, 'val', 'source'))
    targetText = read_single_txt('{}{}.{}'.format(dataDir, 'val', 'target'))
    walk = os.walk(runPath)
    try:  # works if the run contains checkpaths
        _, models, _ = next(walk)
        log('Checkpoints:', models)
        modelPaths = []
        for checkpoint in models:
            modelPaths.append(runPath + '/' + checkpoint)
            predict(modelPaths)
    except StopIteration:  # else just take the run path as model path
        log('no checkpoints')
        modelPaths = [runPath]
    predict(modelPaths)



if __name__ == '__main__':
  fire.Fire(evaluate)
