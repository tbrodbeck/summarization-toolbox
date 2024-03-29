"""
this module provides all
classes and functions to
create a training data set
"""
from typing import Tuple, Union, Optional

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    """dataset to train on

    Args:
        Dataset ([type]): pytorch Dataset
    """

    def __init__(self, encodings: object, labels: torch.Tensor):
        """data set which can be used in a dataloader

        Args:
            encodings (object): tokenized text
            labels (torch.Tensor): tokenized summary
        """
        # transform to the correct data structure
        self.data = [{
            'input_ids': ids,
            'attention_mask': masks,
            'labels': label
        } for ids, masks, label in zip(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels['input_ids']
        )]

        self.n_data = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.n_data


class CustomTokenizer:
    """
    split and tokenize the texts
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split_and_tokenize(self, data: list, ratio: Optional[float] = 0.2) -> tuple:
        """create tokinization for train and val set

        Args:
            data (list): data to tokenize
            ratio (float, optional): train-val-split. Defaults to 0.2.

        Returns:
            tuple: train and validation data
        """
        texts, labels = zip(*data)

        # train test split
        train_set, val_set, train_labels, val_labels = train_test_split(
            texts, labels, test_size=ratio
        )

        # tokinization parameters (required for bart)
        pad_limit = 'max_length'
        trunc_limit = 'longest_first'

        # create sets
        train_set = self.tokenizer(
            train_set, truncation=trunc_limit, padding=pad_limit, return_tensors="pt")
        val_set = self.tokenizer(
            val_set, truncation=trunc_limit, padding=pad_limit, return_tensors="pt")

        train_labels = self.tokenizer(
            train_labels, truncation=trunc_limit, padding=pad_limit, return_tensors="pt")
        val_labels = self.tokenizer(
            val_labels, truncation=trunc_limit, padding=pad_limit, return_tensors="pt")

        return (train_set, train_labels), (val_set, val_labels)

    def tokenize(self, text: Union[str, list]):
        """create token encoding for given texts

        Args:
            text (Union[str, list]): text which is going to be summarized

        Returns:
            [object]: encoded texts in pytoch format for training
        """

        # tokinization parameters (required for bart)
        pad_limit = 512  # 'max_length'
        trunc_limit = 512  # 'longest_first'

        if isinstance(text, str):
            # create sets
            encodings = self.tokenizer(
                text,
                truncation=trunc_limit,
                padding=pad_limit,
                return_tensors="pt"
            )
        else:
            encodings = self.tokenizer(
                text,
                truncation=trunc_limit,
                padding=pad_limit,
                return_tensors="pt"
            )

        return encodings


def create_dataset(
    train_set: Tuple[torch.Tensor, torch.Tensor],
    val_set: Tuple[torch.Tensor, torch.Tensor] = None) \
        -> Tuple[CustomDataset, Union[CustomDataset, None]]:
    """bring data in right shape for training/inference

    Args:
        train_set (Tuple[torch.Tensor, torch.Tensor]): dataset to train on after tokinization
        val_set (Tuple[torch.Tensor, torch.Tensor], optional): validation set for training.
        Defaults to None.

    Returns:
        Tuple[CustomDataset, Union[CustomDataset, None]]: training dataset object
    """
    # create dataset object
    train_data_set = CustomDataset(*train_set)

    if val_set:
        val_data_set = CustomDataset(*val_set)
        return train_data_set, val_data_set

    return train_data_set, None
