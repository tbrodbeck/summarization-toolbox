import fire
from timelogging.timeLog import log
import torch
import transformers
from typing import Dict, List
from .gerneral_io_utils import read_single_txt, write_txt

DATASET_NAMES = ['golem', 'xsum', 'cnn/dailymail']
SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']
TEXT_NAMES = ['source', 'target']

def provideData(datasetName: str, tokenizerName: str, size: int = None, createSplits: Dict = None, splits2tokenize: List = SPLIT_NAMES):
  """Provides tokenized data for training

  Args:
      datasetName (str)
      tokenizerName (str)
      size (int, optional): Defaults to None.
      createSplits (Dict, optional): Split the dataset into train, validation and test splits. Defaults to None.
      splits2tokenize (List, optional): Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.

  Raises:
      ValueError: incorrect inputs"""
  # check input
  if not datasetName in DATASET_NAMES:
    raise ValueError('unkown dataset')
  if not tokenizerName in TOKENIZER_NAMES:
    raise ValueError('unkown tokenizer')
  if size and size < 1:
    raise ValueError('wrong size')

  # retrieve dataset
  if datasetName == 'golem':
    dataDir = 'dataProvider/datasets/golem/'

  if createSplits:
    data = {}
    data['source'] = read_single_txt(dataDir + 'source.txt')
    data['target'] = read_single_txt(dataDir + 'target.txt')
    entries = len(data['source'])
    assert entries == len(data['target']), "Source and target must have the same amount of lines"
    for textName in TEXT_NAMES:
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

  for textName in TEXT_NAMES:  # tokenize
   for splitName in splits2tokenize:
    text = read_single_txt('{}{}.{}'.format(dataDir, splitName, textName))
    if size:
      text = text[:size]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerName)
    log(f'creating batch {textName} for {splitName}')
    pt_batch = tokenizer(
      text,
      padding=True,
      truncation=True,
      return_tensors="pt"
    )
    torch.save(pt_batch, '{}{}_{}.pt'.format(dataDir, splitName, textName))

if __name__ == "__main__":
  fire.Fire(provideData)
