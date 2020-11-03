"""
build an abstractive sumarizer pipeline
using the T5 model
"""

import os
import spacy
from timelogging.timeLog import log
from typing import Dict, Tuple
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
import torch
from utilities.gerneral_io_utils import check_make_dir
from utilities.cleaning_utils import truncate_incomplete_sentences



class AbstractiveSummarizer:
    """
    class for abstractive text summarization
    """

    def __init__(self, language: str, status: str = 'base', base_path: str = './output/results', version: int = None, freezed_layers: list = []):
        """
        define model
        """
        self.base_path = base_path
        if not check_make_dir(self.base_path, create_dir=True):
            log(f"directory {self.base_path} was created")

        self.version = str(version)

        assert language in ["english", "german"], \
        f"{language} is not a supported language!"
        self.language = language

        # init the spacy language model
        # for post processing output
        if language == "english":
            self.nlp = spacy.load("en")
        else:
            self.nlp = spacy.load("de")

        assert status in ['base', 'fine-tuned']
        self.status = status

        # set amount of tokens
        self.max_tokens = 512

        # TODO: make the customized tokens work
        self.model_args, self.tokenizer_args = self.set_args()

        # available models
        # t5: for english texts
        # bart: for german texts
        if language == "english":
            self.model_name = 't5-base'
            self.short_name = 't5'

        elif language == "german":
            self.model_name = 'WikinewsSum/t5-base-multi-de-wiki-news'
            self.short_name = 't5-de'

        # apply custom configurations for model
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            # **self.model_args
        )

        # initialize the model and tokenizer
        # based on parameters
        self.model, self.tokenizer = self.initialize_model()

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

        # optimizer and scheduler only
        # needed for fine-tuning
        self.optimizer = None
        self.scheduler = None


    def set_mode(self, mode: str):
        """
        set model to train or
        evaluation mode
        :param mode:
        :return:
        """
        assert mode in ['eval', 'train'], \
            "Mode can only be 'eval' or 'train'!"

        if mode == "eval":
            self.model.eval()
        else:
            self.model.train()

        log(f"set model to {mode} mode")


    def set_args(self) -> Tuple[Dict, Dict]:
        """
        customize args for tokenizer and model
        :return:
        """
        model_args = {
            # "max_positional_embeddings": self.max_tokens
        }
        tokenizer_args = {
            # "max_len": self.max_tokens,
            # "model_max_length": self.max_tokens,
            # "max_len_sentences_pair": self.max_tokens - 4,
            # "max_len_single_sentence": self.max_tokens - 2
        }

        return model_args, tokenizer_args


    def initialize_model(self):
        """
        check for existing models
        or initialize with raw model
        :return:
        """
        model = None

        if self.status == "fine-tuned":
            if 'output' in self.base_path:
                model_dir = os.path.join(
                        self.base_path,
                        self.short_name
                    )
                assert len(os.listdir(model_dir)) != 0, \
                    f"\nNo pretrained models for {self.short_name}!"

                if check_make_dir(model_dir, create_dir=False):
                    model_path = os.path.join(
                        self.base_path,
                        self.short_name,
                        self.version
                    )
                    if check_make_dir(model_path, create_dir=False):
                        log(f"initialize with model from directory: {model_path}")
                    else:
                        versions = os.listdir(model_dir)
                        latest_version = max(versions)
                        model_path = os.path.join(
                            self.base_path,
                            self.short_name,
                            latest_version
                        )
                        log(f"version {self.version} not found! "
                              f"chose latest version: {latest_version}")

                    model = AutoModelWithLMHead.from_pretrained(model_path)
            else:
                log(f"no pretrained models for {self.short_name}")
                exit()
        else:
            log(f"initialize with base model: {self.short_name}")

            # TODO: Enable loading model from config
            #model = AutoModelWithLMHead.from_config(self.config)
            model = AutoModelWithLMHead.from_pretrained(self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            # **self.tokenizer_args
        )

        return model, tokenizer


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
                    self.model.shared.weight.requires_grad = False
                    self.model.shared.eval()
                if model_component == 'encoder':
                    self.model.encoder.embed_tokens.weight.requires_grad = False
                    self.model.encoder.eval()
                if model_component == 'decoder':
                    self.model.decoder.embed_tokens.weight.requires_grad = False
                    self.model.decoder.eval()
                if model_component == 'lm_head':
                    self.model.lm_head.weight.requires_grad = False
                    self.model.lm_head.eval()
                log(f"freezed {model_component} layers")


    def predict(self, text: str) -> str:
        '''
        predict a summary based on
        the given text
        :param text:
        :return:
        '''

        # tokenize text for model
        text_tokens = self.tokenizer.encode(text, padding="max_length", return_tensors="pt").to(self.device)

        n_tokens = text_tokens.shape[1]

        upper_bound = int(n_tokens * self.upper_token_ratio)
        lower_bound = int(n_tokens * self.lower_token_ratio)

        # produce summary
        summary_ids = self.model.generate(
         text_tokens,
         num_beams=5,
         no_repeat_ngram_size=2,
         min_length=lower_bound,
         max_length=upper_bound,
         early_stopping=True
        )

        # convert the ids to text
        summary_text = self.tokenizer.decode(
         summary_ids[0],
         skip_special_tokens=True
        )
        # remove incomplete sentences
        summary_text = truncate_incomplete_sentences(summary_text, self.nlp)
        # remove leading blanks
        summary_text = summary_text.strip()

        return summary_text
