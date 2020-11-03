import fire
from timelogging.timeLog import log
import torch
import transformers
from typing import Dict, List
from .gerneral_io_utils import read_single_txt, write_txt

DATASET_NAMES = ['golem', 'xsum', 'cnn/dailymail']
MODEL_NAMES = ['t5-base']
SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']
TEXT_NAMES = ['source', 'target']

def provideData(datasetName: str, tokenizerName: str, modelName: str, size: int = None, createSplits: Dict = None, splits2tokenize: List = SPLIT_NAMES):
  """Provides tokenized data for training
  Args:
    datasetName (str)
    tokenizerName (str)
    modelName (str)
    size (int, optional): Defaults to None.
    createSplits (Dict, optional): Split the dataset into train, validation and test splits. Defaults to None.
    splits2tokenize (List, optional): Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.
  Raises:
    ValueError: incorrect inputs"""
  # check input
  if not datasetName in DATASET_NAMES:
    raise ValueError('unkown dataset')
  if not modelName in MODEL_NAMES:
    raise ValueError('unkown model')
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

  # tokenize
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerName)
  maxTokenSize = tokenizer.max_model_input_sizes[modelName]
  for splitName in splits2tokenize:
    textDeleteMask = []
    sourceText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'source'))
    targetText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'target'))
    if size:
      sourceText = sourceText[:size]
      targetText = targetText[:size]
    log(f'tokenizing target batch for {splitName}')
    targetTokens = tokenizer(targetText, padding=True)
    if len(targetTokens['attention_mask'][0]) > maxTokenSize:
      targetTokens = len(targetTokens['attention_mask'][0])
      raise IOError(f'target contains more than {maxTokenSize} tokens: {targetTokens}')
    log(f'tokenizing source batch for {splitName}')
    sourceTokens = tokenizer(sourceText, padding=True, max_length=maxTokenSize + 1)
    for i, attention in enumerate(sourceTokens['attention_mask']):
      if len(attention) < maxTokenSize:
        break
      if attention[maxTokenSize]:
        textDeleteMask.append(i)
    for textName, tokens in [('source', sourceTokens), ('target', targetTokens)]:
      shortTokens = {}
      for key in tokens:
        tokensList = tokens[key]
        for i in sorted(textDeleteMask, reverse=True):
          del tokensList[i]
        tokensTensor = torch.LongTensor(tokensList[:maxTokenSize])
        shortTokens[key] = tokensTensor
      torch.save(shortTokens, '{}{}_{}.pt'.format(dataDir, splitName, textName))

if __name__ == "__main__":
  fire.Fire(provideData)
