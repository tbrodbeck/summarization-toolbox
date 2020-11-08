import sys
sys.path.append(".")

from timelogging.timeLog import log
import torch
import transformers
from typing import List
import fire
from utilities.gerneral_io_utils import read_single_txt, write_txt, assertDirExistent, assertFileInxestent, check_make_dir
from utilities.parser_utils import parser

SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']

def provideData(datasetName: str, tokenizerName: str, size: int = None, createSplits=None, splits2tokenize: List = SPLIT_NAMES):
  """Provides tokenized data for training
  Args:
    datasetName (str)
    tokenizerName (str)
    size (int, optional): Defaults to None.
    createSplits (Dict, optional): Split the dataset into train, validation and test splits. Defaults to None.
                                   Has to be provided as a dict containing the keys `train` and `val` and values
                                   between 0 and 1. If `True` uses a default 80/10/10 split.
    splits2tokenize (List, optional): Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.
  Raises:
    ValueError: incorrect inputs"""

  # checking input
  if not tokenizerName in TOKENIZER_NAMES:
    raise ValueError('unkown tokenizer')
  if size and size < 1:
    raise ValueError('wrong size')

  # connecting to dataset
  dataDir = f'dataProvider/datasets/{datasetName}/'
  assertDirExistent(dataDir)

  if createSplits:
    if createSplits == True:
      createSplits = {'train': 0.8, 'val': 0.5}
    for splitKey in createSplits:
      if not splitKey in SPLIT_NAMES:
        raise ValueError(f'unkown key {splitKey} - createSplits has to be a dictionary containing the keys `train` and `val` and values between 0 and 1')
    data = {}
    data['source'] = read_single_txt(dataDir + 'sources.txt')
    data['target'] = read_single_txt(dataDir + 'targets.txt')
    entries = len(data['source'])
    assert entries == len(data['target']), "Source and target must have the same amount of lines"
    for textName in ['source', 'target']:
      text = data[textName]
      previousSplitIndex = 0
      createSplits['test'] = 1.
      for splitName in SPLIT_NAMES:
        splitFraction = createSplits[splitName]
        if not 0 <= splitFraction <= 1:  # check split values
          raise ValueError('incorrect split sizes')
        splitIndex = int((entries - previousSplitIndex) * splitFraction + previousSplitIndex)
        split = text[previousSplitIndex:splitIndex]
        write_txt('{}{}.{}'.format(dataDir, splitName, textName), split)
        previousSplitIndex = splitIndex
      assert previousSplitIndex == entries, f'{previousSplitIndex} != {entries}'

  # tokenizing
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerName)

  tensorDir = f'{dataDir}tensors/'
  check_make_dir(tensorDir, True)

  for splitName in splits2tokenize:
    sourceText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'source'))
    targetText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'target'))
    if size:
      sourceText = sourceText[:size]
      targetText = targetText[:size]

    pad_limit = 'max_length'
    trunc_limit = 'longest_first'

    log(f'tokenizing target batch for {splitName}')
    targetTokens = tokenizer(targetText, padding=pad_limit, truncation=trunc_limit, return_tensors='pt')
    log(f'tokenizing source batch for {splitName}')
    sourceTokens = tokenizer(sourceText, padding=pad_limit, truncation=trunc_limit, return_tensors='pt')

    for textName, tokens in [('source', sourceTokens), ('target', targetTokens)]:
      tensorPath = f'{tensorDir}{splitName}_{textName}.pt'
      assertFileInxestent(tensorPath)
      torch.save(tokens, tensorPath)


if __name__ == "__main__":
  fire.Fire(provideData)

