"""
build an abstractive sumarizer pipeline
using the T5 model
"""

import os
import spacy
from transformers import AutoModelWithLMHead, AutoTokenizer, BatchEncoding
import torch
from timelogging.timeLog import log
from typing import Union
from utilities.gerneral_io_utils import check_make_dir
from utilities.cleaning_utils import truncate_incomplete_sentences


class AbstractiveSummarizer:
    """
    class for abstractive text summarization
    """

    def __init__(self, language: str, status: str = 'base', model_dir: str = './results', version: int = None, freezed_layers: list = None, checkpoint: int = None):
        """
        define model
        """
        assert language in ["english", "german"], \
            f"{language} is not a supported language!"
        self.language = language
        # available models
        # t5: for english texts
        # bart: for german texts
        if language == "english":
            self.model_name = 't5-base'
            self.short_name = 't5'

        elif language == "german":
            self.model_name = 'WikinewsSum/t5-base-multi-de-wiki-news'
            self.short_name = 't5-de'

        assert status in ['base', 'fine-tuned']
        self.status = status
        if self.status == 'base':
            log(f"You chose status '{self.status}'. "
                f"Pre-trained '{self.model_name}' is loaded.")
        else:
            if checkpoint:
                self.model_path = os.path.join(
                    model_dir,
                    self.short_name,
                    str(version),
                    f"checkpoint-{checkpoint}"
                )
            else:
                self.model_path = os.path.join(
                    model_dir,
                    self.short_name,
                    str(version)
                )
            assert check_make_dir(self.model_path), \
                f"Directory '{self.model_path}' doesn't exist! Please follow this folder structure."
            log(f"You chose status '{self.status}'. "
                f"Fine-tuned '{self.model_name}' from directory '{self.model_path}' is loaded.")

        # initialize the model and tokenizer
        # based on parameters
        self.model, self.tokenizer = self.initialize_model()

        # init the spacy language model
        # for post processing output
        if language == "english":
            self.nlp = spacy.load("en")
        else:
            self.nlp = spacy.load("de")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        log(f"{self.device} available")

        # freeze layers not to train
        if freezed_layers:
            self.freeze_model_layers(freezed_layers)

        # upper and lower bound for produced
        # summary (found by data analysis)
        self.upper_token_ratio = 0.15
        self.lower_token_ratio = 0.05


    def initialize_model(self):
        """
        check for existing models
        or initialize with raw model
        :return:
        """
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "training_args.bin"
        ]
        if self.status == 'base':
            return AutoModelWithLMHead.from_pretrained(self.model_name), \
                   AutoTokenizer.from_pretrained(self.model_name)
        else:
            assert all([file in os.listdir(self.model_path) for file in required_files]), \
                f"Not all required files {'/'.join(required_files)} in {self.model_path}!"
            return AutoModelWithLMHead.from_pretrained(self.model_path), \
                   AutoTokenizer.from_pretrained(self.model_name)


    @staticmethod
    def freeze_params(component):
        for par in component.parameters():
            par.requires_grad = False


    def freeze_model_layers(self, layers: list):
        """
        freeze layers
        :return:
        """
        model_layers = [
            "shared",
            "encoder",
            "decoder",
            "lm_head"
        ]

        # this is special freezing for T5
        if all([layer in model_layers for layer in layers]):
            for model_component in layers:
                if model_component == 'shared':
                    self.freeze_params(self.model.shared)
                if model_component == 'encoder':
                    self.freeze_params(self.model.encoder.embed_tokens)
                if model_component == 'decoder':
                    self.freeze_params(self.model.decoder.embed_tokens)
                if model_component == 'lm_head':
                    self.freeze_params(self.model.model.lm_head)
                log(f"freezed {model_component} layers")


    def predict(self, source: Union[str, dict], truncation: bool = True) -> Union[list, str]:
        '''
        predict a summary based on
        the given text
        :param truncation:
        :param source:

        :return:
        '''
        if isinstance(source, dict):
            model_inputs = list()
            n_tokens = list()
            for ids in source['input_ids']:
                model_inputs.append(
                    ids.unsqueeze(0)
                )
                n_tokens.append(len([i for i in ids if i != 0]))

            return_string = False
        elif isinstance(source, str):
            # tokenize text for model
            model_inputs = [self.tokenizer.encode(source, padding="max_length", truncation="longest_first", return_tensors="pt").to(self.device)]
            n_tokens = [len([i for i in model_inputs[0]['input_ids'] if i != 0])]

            return_string = True
        else:
            raise ValueError("Input to 'predict' has to be str or dict!")

        upper_bounds = [int(n * self.upper_token_ratio) for n in n_tokens]
        lower_bounds = [int(n * self.lower_token_ratio) for n in n_tokens]

        # produce summary
        summary_texts = list()
        for tokens, upper_bound, lower_bound in zip(model_inputs, upper_bounds, lower_bounds):
            summary_ids = self.model.generate(
             tokens,
             num_beams=5,
             no_repeat_ngram_size=2,
             min_length=lower_bound,
             max_length=upper_bound,
             early_stopping=True
            ).to(self.device)

            # convert the ids to text
            summary_text = self.tokenizer.decode(
             summary_ids[0],
             skip_special_tokens=True
            )

            if truncation:
                # remove incomplete sentences
                summary_text = truncate_incomplete_sentences(summary_text, self.nlp)
                # remove leading blanks
                summary_text = summary_text.strip()

            summary_texts.append(
                summary_text
            )

        if return_string:
            return summary_texts[0]
        return summary_texts
