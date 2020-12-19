import sys
sys.path.append(".")
from . import metrics
from os.path import join
from timelogging.timeLog import log
import torch
from typing import Dict
from utilities import io_utils

class SemanticSimilarityCreator():
  def __init__(self, language: str):
    self.metric = metrics.SemanticSimilarityMetric(language)

  def createNewTextSnipped(self, text, embeddings) -> str:
    textSnippet = text[:100]
    return textSnippet

  def embedTextFile(self, dataSubsetPath) -> Dict[str, torch.Tensor]:
    embeddings = {}
    texts = io_utils.read_single_txt(dataSubsetPath)
    log(f'embedding the {dataSubsetPath} containing {len(texts)} entries')
    for text in texts:
      # textSnippet = self.createNewTextSnipped(text, embeddings) # optional if embeddings get too large because of the strings; then replace text with textSnippet as key in the dictionary
      embeddings[text] = self.metric.create_embedding(text)
    return embeddings

  def embedDataSubsetOfDatatype(self, datasetPath: str, subsetType: str, dataType: str) -> Dict[str, torch.Tensor]:
    return self.embedTextFile(join(datasetPath, f'{subsetType}.{dataType}'))

  def saveEmbeddings(self, datasetPath: str, subsetType: str, savePath: str):
    self.saveEmbeddingsOfDatatype(datasetPath, subsetType, 'source', savePath)
    self.saveEmbeddingsOfDatatype(datasetPath, subsetType, 'target', savePath)
    log(f'sucessfully saved sources and targets of {subsetType} to {savePath}')

  def saveEmbeddingsOfDatatype(self, datasetPath: str, subsetType: str, dataType: str, savePath: str):
    saveFilePath = join(savePath, f'{subsetType}_{dataType}.pt')
    embeddings = self.embedDataSubsetOfDatatype(datasetPath, subsetType, dataType)
    io_utils.check_make_dir(savePath, create_dir=True)
    torch.save(embeddings, open(saveFilePath, 'wb'))
