"""
build an abstractive sumarizer pipeline
using the T5 model
"""

import os
from typing import Union, Optional, Tuple
from timelogging.timeLogLater import logOnce as log
import spacy
from transformers import AutoModelWithLMHead, AutoTokenizer, AdamW
import torch
from utilities.io_utils import check_make_dir
from utilities.cleaning_utils import truncate_incomplete_sentences


class AbstractiveSummarizer:
    """
    class for abstractive text summarization
    """

    def __init__(
            self,
            model_dir: str,
            language: str,
            status: Optional[str] = 'base'):
        """set arguments to initialize the model used for summarization

        Args:
            model_dir (str): direction to load/store model
            language (str): supported language
            status (Optional[str], optional): sets if model is
            already fine-tuned or not. Defaults to 'base'.
        """
        self.model_path = model_dir

        assert language in ["english", "german"], \
            f"{language} is not a supported language!"
        self.language = language
        # available models
        # t5-base: for english texts
        # tWikinewsSum/t5-base-multi-de-wiki-news: for german texts
        if language == "english":
            self.model_name = 't5-base'
            self.short_name = 't5'

        elif language == "german":
            self.model_name = 'WikinewsSum/t5-base-multi-de-wiki-news'
            self.short_name = 't5-de'

        assert status in ['base', 'fine-tuned']
        self.status = status
        if self.status != 'base':
            assert check_make_dir(self.model_path), \
                f"Directory '{self.model_path}' doesn't exist! \
                    Please follow this folder structure."

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

    def initialize_model(self) -> Tuple[AutoModelWithLMHead, AutoTokenizer]:
        """check for existing models or initialize with raw model

        Returns:
            Tuple[AutoModelWithLMHead, AutoTokenizer]: model and tokenizer
            used for training and inference
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


    def freeze_params(self, component: object):
        """make model component un-trainable

        Args:
            component (object): part of transformer model
        """
        for par in component.parameters():
            par.requires_grad = False

    def freeze_model_layers(self, layers: list):
        """define layers that should not be trained

        Args:
            layers (list): names of layers contained by transformer
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
                    self.freeze_params(self.model.lm_head)

    def predict(self,
                source: Union[str, dict],
                truncation: bool = True,
                upper_token_ratio: float = 0.15,
                lower_token_ratio: float = 0.05) -> Union[list, str]:
        """inference from the summary model based on given texts

        Args:
            source (Union[str, dict]): text to summarize
            truncation (bool, optional): truncate the given text if too long.
            Defaults to True.
            upper_token_ratio (float, optional): set the upper bound for the summary.
            Defaults to 0.15.
            lower_token_ratio (float, optional): set the lower bound for the summary.
            Defaults to 0.05.

        Raises:
            ValueError: wrong input format

        Returns:
            Union[list, str]: returns batch of summaries
            or one summary string
        """
        # set model to evaluation
        # and choose gpu if available
        self.model.eval()
        self.model.to(self.device)

        # extract tokens for inference
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
            model_inputs = [self.tokenizer(
                source,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt").to(self.device)['input_ids']]
            n_tokens = [
                len([i for i in model_inputs[0].squeeze().detach().cpu().numpy() if i != 0])]

            return_string = True
        else:
            raise ValueError("Input to 'predict' \
                has to be str or dict!")

        # set length of outcoming summary
        upper_bounds = [int(n * upper_token_ratio) for n in n_tokens]
        lower_bounds = [int(n * lower_token_ratio) for n in n_tokens]

        # produce summary
        summary_texts = list()
        try:
            for tokens, upper_bound, lower_bound in \
                    zip(model_inputs, upper_bounds, lower_bounds):
                summary_ids = self.model.generate(
                    tokens.to(self.device),
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
                    summary_text = truncate_incomplete_sentences(
                        summary_text, self.nlp)
                    # remove leading blanks
                    summary_text = summary_text.strip()

                summary_texts.append(
                    summary_text
                )
        except:
            log("A problem producing the summary occured!")
            summary_texts.append("")
            

        # outcome dependent on input format
        if return_string:
            if summary_texts[0] == "":
                log("Input text too short for summarization!")
            return summary_texts[0]
        return summary_texts
