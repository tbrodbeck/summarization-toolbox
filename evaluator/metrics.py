"""
module for all supported metrics
"""
import sentence_transformers
import torch


class Metric:
    """Interface for Metrics"""

    def get_score(self, prediction: str, target: str) -> float:
        """dummy get score

        Args:
            prediction (str):
            target (str):

        Returns:
            float:
        """
        return 0.0


class SemanticSimilarityMetric(Metric):
    """
    calculates cosine similarity
    of word embeddings
    """

    def __init__(self, language: str):
        """initializes metric for language

        Args:
            language (str): german or english
        """

        assert language in ["english", "german"]
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.transformer = sentence_transformers.SentenceTransformer(
            "bert-base-nli-stsb-mean-tokens"
        ).to(self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        if language == "english":
            print("Intitialized semantic similarity metric for English texts.")

        else:
            print("Intitialized semantic similarity metric for German texts.")




    def get_score(self, prediction: str, target: str) -> float:
        """calculates score for the semantic similarity metric
           higher score -> higher semanctic similarity

        Args:
            prediction (str): predicted summary
            target (str): ground truth

        Returns:
            float: similarity score
        """

        prediction_embeddings = torch.tensor(
            self.transformer.encode(prediction)).unsqueeze(0)
        target_embeddings = torch.tensor(
            self.transformer.encode(target)).unsqueeze(0)

        score = self.cosine_similarity(
            prediction_embeddings, target_embeddings)

        return score.item()
    
    def get_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor(self.transformer.encode(text))

    def get_similarity(self, vector_1: torch.Tensor, vector_2: torch.Tensor) -> float:
        return self.cosine_similarity(vector_1.unsqueeze(0), vector_2.unsqueeze(0)).item()

