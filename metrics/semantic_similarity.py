"""
define metrics for evaluation of predictions
"""
from utilities.gerneral_io_utils import write_excel
from transformers import BertTokenizer, BertModel
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SemanticSimilarityMetric:
    """
    calculates cosine similarity
    of word embeddings
    """
    def __init__(self, language: str):

        assert language in ["en", "de"]

        self.transformer = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)

        if language == "en":
            print("Intitialized semantic similarity metric for English texts.")
        else:
            print("Intitialized semantic similarity metric for German texts.")


    def get_score(self, prediction: str, target:str) -> float:
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

        for text, target in tqdm(zip(texts, targets)):
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

        write_excel(overview_dict, output_path, "evaluation.xlsx")





def semantic_similarity_metric(prediction: str, target: str, lang: str = "en") -> float:
    """
    evaluate based on sentence similarity
    :param lang:
    :param prediction:
    :param target:
    :return:
    """
    cos = CosineSimilarity(dim=1, eps=1e-6)

    if lang == "en":
        language_model = "bert-base-uncased"
    elif lang == "de":
        language_model = "bert-base-german-cased"
    else:
        print("Language not supported!")
        exit()

    # load bert language model
    tokenizer = BertTokenizer.from_pretrained(language_model)
    model = BertModel.from_pretrained(language_model, output_hidden_states=True)

    # put it in evaluation mode
    model.eval()

    # encode text to bert tokens
    encoding = tokenizer.batch_encode_plus([prediction, target])

    # extract needed input for models
    ids = [torch.tensor(item).unsqueeze(0) for item in encoding['input_ids']]
    sequence = [torch.tensor(item).unsqueeze(0) for item in encoding['attention_mask']]

    # retrieve sentence embeddings
    with torch.no_grad():
        mean_embeddings = list()
        for i in range(2):
            outputs = model(ids[i], sequence[i])
            # get hidden model states
            hidden_states = outputs[2]
            # these are the embedding layers
            token_vecs = hidden_states[-2][0]
            # calculate mean embeddings for layers
            mean_embeddings.append(
                torch.mean(token_vecs, dim=0).unsqueeze(0)
            )

    # evaluate the distance between two embeddings
    # by cosine similarity
    similarity_score = cos(*mean_embeddings)

    return float(similarity_score)


# module for testing the score
if __name__ == '__main__':
    score = semantic_similarity_metric(
        "Football is my favourite hobby.",
        "I love to play soccer."
    )
    print("Score:", score)