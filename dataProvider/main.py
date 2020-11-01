import os
from timelogging.timeLog import log
import typing

DATASET_NAMES = ['golem', 'xsum', 'cnn/dailymail']
TOKENIZER_NAMES = ['t5Tokenizer']

class DataProvider():
  def __init__(self, datasetName: str, tokenizerName: str, size: int = None, createSplits: typing.Tuple = None):
    # check input
    if not datasetName in DATASET_NAMES:
      raise ValueError('unkown dataset')
    if not tokenizerName in TOKENIZER_NAMES:
      raise ValueError('unkown tokenizer')
    if size and size < 1:
      raise ValueError('wrong size')
    if datasetName == 'golem':
      dataDir = 'dataProvider/datasets/golem/'
    if createSplits:
      sourceFile = dataDir + 'source.txt'
      targetFile = dataDir + 'target.txt'
      if not os.path.isfile(sourceFile) or not os.path.isfile(targetFile):  # check if correct files exist
        raise ValueError(f'{sourceFile} or {targetFile} not found')
      splitNames = ['train', 'val', 'test']
      for i, splitFraction in enumerate(createSplits):
        if not 0 <= splitFraction <= 1:  # check split values
          raise ValueError('incorrect split sizes')
        sourceFile = dataDir + '{}.source'.format(splitNames[i])
        targetFile = dataDir + '{}.target'.format(splitNames[i])
        if os.path.isfile(sourceFile) or os.path.isfile(targetFile):  # check if correct files are not existent yet
          raise ValueError(f'{sourceFile} or {targetFile} not found')

  def read_single_txt(file_path: str, limit: int = None) -> typing.List[str]:
    """
    read text/lines from
    a single text file
    :param limit:
    :param file_path:
    :return:
    """
    lines = list()
    log("\nRead", file_path)
    with open(file_path, mode="r", encoding="utf-8") as file_handle:
        text = file_handle.readlines()
        lines = [line.rstrip('\n') for line in text]

    if limit:
        lines = lines[:limit]

    return lines
