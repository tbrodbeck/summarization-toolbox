import fire
import os
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from transformers import AutoConfig, AutoModelWithLMHead, TrainingArguments, Trainer

def evaluate(dataPath):
  walk = os.walk(dataPath)
  _, checkpoints, _ = next(walk)
  print(checkpoints)  # TODO read and evaluate checkpoints
  summarizer = AbstractiveSummarizer(
    language="german",
    status="fine-tuned",
    version=0,
    checkpoint = 500
  )
  print("Initialized summarizer!")

if __name__ == '__main__':
  #fire.Fire(evaluate)
  config = AutoConfig.from_pretrained("./results/t5-de/1/checkpoint-500")
  model = AutoModelWithLMHead.from_pretrained("./results/t5-de/1/checkpoint-500", force_download=True)
  print(config)
