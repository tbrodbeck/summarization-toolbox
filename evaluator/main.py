import fire
import os

def evaluate(dataPath):
  walk = os.walk(dataPath)
  _, checkpoints, _ = next(walk)
  print(checkpoints)  # TODO read and evaluate checkpoints

if __name__ == '__main__':
  fire.Fire(evaluate)
