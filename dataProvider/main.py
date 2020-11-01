import timelogging

DATASET_NAMES = ['golem', 'xsum', 'cnn/dailymail']
TOKENIZER_NAMES = ['t5Tokenizer']

class DataProvider():
  def __init__(self, datasetName: str, tokenizerName: str, size: int = None):
    if not datasetName in DATASET_NAMES:
      raise ValueError('unkown dataset')
    if not tokenizerName in TOKENIZER_NAMES:
      raise ValueError('unkown tokenizer')
    if size and size < 1:
      raise ValueError('wrong size')
