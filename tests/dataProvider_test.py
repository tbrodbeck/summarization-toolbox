import dataProvider.main as o
import unittest

def p(modulePath):
    return 'main.' + modulePath

class TestDataProvider(unittest.TestCase):
  def test_provideData(self):
    # # all possible combinations can be created
    # for datasetName in o.DATASET_NAMES:
    #   for tokenizerName in o.TOKENIZER_NAMES:
    #     o.provideData(datasetName, tokenizerName)

    self.assertRaises(ValueError, o.provide_data, o.TOKENIZER_NAMES[0], 'dataset', o.MODEL_NAMES[0], 1)

    # no impossible input
    self.assertRaises(ValueError, o.provide_data, 'unknown', 'dataset', o.MODEL_NAMES[0])
    self.assertRaises(ValueError, o.provide_data, o.TOKENIZER_NAMES[0], 'unknown', o.MODEL_NAMES[0])
    self.assertRaises(ValueError, o.provide_data, o.TOKENIZER_NAMES[0], 'dataset', 'unknown')
    self.assertRaises(ValueError, o.provide_data, o.TOKENIZER_NAMES[0], 'dataset', -1)

    # dataset = 'golem'
    # # dataPath = f'dataProvider/datasets/{dataset}/'
    # # filesToDelete = glob.glob(f'{dataPath}*.source') + glob.glob(f'{dataPath}*.target')
    # # for filePath in filesToDelete:
    # #   os.remove(filePath)
    # # o.provideData(dataset, o.TOKENIZER_NAMES[0], o.MODEL_NAMES[0], createSplits={'train': 0.8, 'val': 0.5}, size=100)
    # o.provideData(dataset, o.TOKENIZER_NAMES[0], o.MODEL_NAMES[0], size=100)
