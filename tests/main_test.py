import dataProvider.main as o
import unittest

def p(modulePath):
    return 'main.' + modulePath

class TestDataProvider(unittest.TestCase):
  def testMain(self):
    # all possible combinations can be created
    for datasetName in o.DATASET_NAMES:
      for tokenizerName in o.TOKENIZER_NAMES:
        o.DataProvider(datasetName, tokenizerName)
    self.assertRaises(ValueError, o.DataProvider, o.TOKENIZER_NAMES[0], o.DATASET_NAMES[0], 1)

    # no impossible input
    self.assertRaises(ValueError, o.DataProvider, 'unknown', o.DATASET_NAMES[0])
    self.assertRaises(ValueError, o.DataProvider, o.TOKENIZER_NAMES[0], 'unknown')
    self.assertRaises(ValueError, o.DataProvider, o.TOKENIZER_NAMES[0], o.DATASET_NAMES[0], -1)
