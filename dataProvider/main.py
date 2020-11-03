import fire
from timelogging.timeLog import log
import torch
import transformers
from typing import Dict, List
from .gerneral_io_utils import assertDirExistent, assertFileInxestent, check_make_dir, read_single_txt, write_txt

MODEL_NAMES = ['t5-base']
SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']

def provideData(datasetName: str, tokenizerName: str, modelName: str, size: int = None, createSplits: Dict = None, splits2tokenize: List = SPLIT_NAMES):
  """Provides tokenized data for training
  Args:
    datasetName (str)
    tokenizerName (str)
    modelName (str)
    size (int, optional): Defaults to None.
    createSplits (Dict, optional): Split the dataset into train, validation and test splits. Defaults to None. Has to be provided as a dict containing the keys `train` and `val` and values between 0 and 1. If `True` uses a default 80/10/10 split.
    splits2tokenize (List, optional): Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.
  Raises:
    ValueError: incorrect inputs"""
  # checking input
  if not modelName in MODEL_NAMES:
    raise ValueError('unkown model')
  if not tokenizerName in TOKENIZER_NAMES:
    raise ValueError('unkown tokenizer')
  if size and size < 1:
    raise ValueError('wrong size')

  # connecting to dataset
  dataDir = f'dataProvider/datasets/{datasetName}/'
  assertDirExistent(dataDir)

  if createSplits:
    # if
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
  maxTokenSize = tokenizer.max_model_input_sizes[modelName]
  tensorDir = f'{dataDir}tensors/'
  check_make_dir(tensorDir, True)
  for splitName in splits2tokenize:
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
    # finding tokenizations that are too long
    tokensDeletes = []
    for i, attention in enumerate(sourceTokens['attention_mask']):
      if len(attention) < maxTokenSize:
        break
      if attention[maxTokenSize]:
        tokensDeletes.append(i)
    # filtering and saving to pt tensor
    for textName, tokens in [('source', sourceTokens), ('target', targetTokens)]:
      shortTokens = {}
      for key in tokens:
        tokensList = tokens[key]
        for i in sorted(tokensDeletes, reverse=True):  # filtering
          del tokensList[i]
        tokensTensor = torch.LongTensor(tokensList[:maxTokenSize])
        shortTokens[key] = tokensTensor
      tensorPath = f'{tensorDir}{splitName}_{textName}.pt'
      assertFileInxestent(tensorPath)
      torch.save(shortTokens, tensorPath)

if __name__ == "__main__":
  fire.Fire(provideData)
