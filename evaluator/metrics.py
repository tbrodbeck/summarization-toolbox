"""
module for all supported metrics
"""
import sys
sys.path.append(".")
from evaluator.PyRouge.PyRouge import pyrouge
import os
import sentence_transformers
import torch

class Metric:
    """Interface for Metrics"""
    def get_score(self, text1: str, text2: str) -> float:
        return 0.0

class Rouge(Metric):
    def __init__(self):
        self.r = pyrouge.Rouge()

    def get_score(self, text1: str, text2: str) -> float:
        [_, _, f_score] = self.r.rouge_l([text1], [text2])
        return f_score

class SemanticSimilarityMetric(Metric):
    """
    calculates cosine similarity
    of word embeddings
    """

    def __init__(self, language: str, embeddings_path='', dataset_split='val'):
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
        self.saved_embeddings = self.init_embeddings(embeddings_path, dataset_split)

    def init_embeddings(self, embeddings_path: str, dataset_split: str):
        if embeddings_path:
            source_embeddings = self.load_embeddings(embeddings_path, dataset_split, "source")
            target_embeddings = self.load_embeddings(embeddings_path, dataset_split, "target")
            return {**source_embeddings, **target_embeddings}
        else:
            return {}

    def load_embeddings(self, embeddings_path: str, dataset_split: str, dataset_type: str):
        load_file_path = os.path.join(embeddings_path, f"{dataset_split}_{dataset_type}.pt")
        return torch.load(load_file_path)

    def get_score(self, text1: str, text2: str) -> float:
        """calculates score for the semantic similarity metric
           higher score -> higher semanctic similarity
        Returns:
            float: similarity score
        """
        text1_embeddings = self.get_embedding(text1)
        text2_embeddings = self.get_embedding(text2)

        score = self.cosine_similarity(
            text1_embeddings.unsqueeze(0), text2_embeddings.unsqueeze(0))

        return score.item()

    def get_embedding(self, text):
        if text in self.saved_embeddings.keys():
            return self.saved_embeddings[text]
        else:
            return torch.tensor(self.transformer.encode(text, show_progress_bar=False))
    def create_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor(self.transformer.encode(text, show_progress_bar=False))
