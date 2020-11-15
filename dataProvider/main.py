import sys
sys.path.append(".")
import fire
import numpy as np
from timelogging.timeLog import log
import torch
import transformers
import typing
from utilities.gerneral_io_utils import read_single_txt, write_txt, assertDirExistent, assertFileInxestent, check_make_dir

MODEL_NAMES = ['t5-base']
SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']

def provideData(datasetName: str, tokenizerName: str, modelName: str, size: int = None, createSplits=False, splits2tokenize: typing.List = SPLIT_NAMES, filtering=True):
  """Provides tokenized data for training
  Args:
    datasetName (str)
    tokenizerName (str)
    modelName (str)
    size (int, optional): Limits the amount of samples that are taken for tokenization for each split. Defaults to None.
    createSplits (Dict or bool, optional): Split the dataset into train, validation and test splits. Has to be provided as a dict containing the keys `train` and `val` and values between 0 and 1. If `True` uses a default 80/10/10 split. Defaults to False.
    splits2tokenize (List, optional): Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.
    filtering (bool, optional): Longer examples than the maximum token size are filtered, else they are truncated. Defaults to True.
  Raises:
    ValueError: incorrect inputs"""
  # checking input
  if not modelName in MODEL_NAMES:
    raise ValueError('unkown model')
  if not tokenizerName in TOKENIZER_NAMES:
    raise ValueError('unkown tokenizer')
  if size and size < 1:
    raise ValueError('wrong size')
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
  maxTokenSize = tokenizer.max_model_input_sizes[modelName]
  if filtering:
    filtered = '_filtered'
  else:
    filtered = ''
  tensorDir = f'{dataDir}{tokenizerName}{filtered}/'
  check_make_dir(tensorDir, True)
  for splitName in splits2tokenize:
    sourceText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'source'))
    targetText = read_single_txt('{}{}.{}'.format(dataDir, splitName, 'target'))
    textLen = len(sourceText)
    assert textLen == len(targetText)
    if size:  # optional limitation of samples for tokenization
      sourceText = sourceText[:size]
      targetText = targetText[:size]
    log(f'tokenizing target batch for {splitName} of {textLen} samples')
    if filtering:
      targetTokens = tokenizer(targetText, padding=True)
    else:
      targetTokens = tokenizer(targetText, padding=True, return_tensors="pt")
    if len(targetTokens['attention_mask'][0]) > maxTokenSize:
      targetTokens = len(targetTokens['attention_mask'][0])
      raise IOError(f'target contains more than {maxTokenSize} tokens: {targetTokens}')
    log(f'tokenizing source batch for {splitName}')
    if filtering:
      sourceTokens = tokenizer(sourceText, padding='max_length', truncation=True, max_length=maxTokenSize + 1)
    else:
      sourceTokens = tokenizer(sourceText, padding='max_length', truncation=True, return_tensors='pt')
    if filtering:  # finding tokenizations that are too long
      tokensDeletes = []
      for i, attention in enumerate(sourceTokens['attention_mask']):
        if len(attention) < maxTokenSize:
          break
        if attention[maxTokenSize]:
          tokensDeletes.append(i)
      deletedSamples = len(tokensDeletes)
      log('{} ({}%) of samples were filtered because they were too long'.format(deletedSamples, round((deletedSamples / len(sourceTokens['attention_mask'])) * 100, 2)))
    for textName, tokens in [('source', sourceTokens), ('target', targetTokens)]:
      if filtering:  # creating filtered PyTorch tensors from tokenization lists and replacing them
        for key in tokens:  # tokens contains `inputs_ids` and `attention_mask`
          tokensList = tokens[key]
          for i in sorted(tokensDeletes, reverse=True):  # actual filtering
            del tokensList[i]
          tokensTensor = torch.LongTensor(np.array(tokensList)[:, :512])
          tokens[key] = tokensTensor
      tensorPath = f'{tensorDir}{splitName}_{textName}.pt'
      log(f'{tensorPath} with output size:', tokens[list(tokens.keys())[0]].size())
      assertFileInxestent(tensorPath)
      torch.save(tokens, tensorPath)

if __name__ == "__main__":
  fire.Fire(provideData)
