"""
module for all supported metrics
"""
import sentence_transformers
import torch


class SemanticSimilarityMetric:
    """
    calculates cosine similarity
    of word embeddings
    """
    def __init__(self, language: str):

        assert language in ["english", "german"]

        self.transformer = sentence_transformers.SentenceTransformer("bert-base-nli-stsb-mean-tokens")
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        if language == "english":
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

