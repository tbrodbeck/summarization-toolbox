"""
module to read dataset and bring it
in the right format for training
"""
from utilities.general_io_utils import read_single_txt, write_txt, \
  assertDirExistent, assertFileInxestent, check_make_dir
from typing import Optional
import transformers
from timelogging.timeLog import log
import torch
import numpy as np
import fire
import sys
sys.path.append(".")

MODEL_NAMES = ['t5-base']
SPLIT_NAMES = ['train', 'val', 'test']
TOKENIZER_NAMES = ['WikinewsSum/t5-base-multi-de-wiki-news']


def provide_data(
        dataset_name: str,
        tokenizer_name: str,
        model_name: str,
        size: Optional[int] = None,
        create_splits: Optional[bool] = False,
        splits2tokenize: Optional[list] = SPLIT_NAMES,
        filtering: Optional[bool] = True):
    """Provides tokenized data for training

    Args:
        dataset_name (str): foldername in datasets directory
        tokenizer_name (str): huggingface tokenizer name (same as model name)
        model_name (str): huggingface model name
        size (Optional[int], optional): Limits the amount of samples that are
        taken for tokenization for each split.
        create_splits (Optional[bool], optional): Split the dataset into train,
        validation and test splits.
        Has to be provided as a dict containing the keys `train` and `val` and
        values between 0 and 1.
        If `True` uses a default 80/10/10 split. Defaults to False.
        splits2tokenize (Optional[list], optional): Can be set to only
        tokenize certain splits. Defaults to SPLIT_NAMES.
        filtering (Optional[bool], optional): Longer examples than the
        maximum token size are filtered,
        else they are truncated. Defaults to True.

    Raises:
        ValueError: incorrect inputs
        IOError: incompatible text and summary number
    """
    # checking input
    if not model_name in MODEL_NAMES:
        raise ValueError('unkown model')
    if not tokenizer_name in TOKENIZER_NAMES:
        raise ValueError('unkown tokenizer')
    if size and size < 1:
        raise ValueError('wrong size')
    dataset_dir = f'dataProvider/datasets/{dataset_name}/'
    assertDirExistent(dataset_name)

    if create_splits:
        if create_splits == True:
            create_splits = {'train': 0.8, 'val': 0.5}
        for split_key in create_splits:
            if not split_key in SPLIT_NAMES:
                raise ValueError(
                    f'unkown key {split_key} - createSplits has to be a \
                      dictionary containing the keys `train` and `val` \
                        and values between 0 and 1')
        data = {}
        data['source'] = read_single_txt(dataset_dir + 'sources.txt')
        data['target'] = read_single_txt(dataset_dir + 'targets.txt')
        entries = len(data['source'])
        assert entries == len(
            data['target']), "Source and target must have the same amount of lines"
        for text_name in ['source', 'target']:
            text = data[text_name]
            previous_split_index = 0
            create_splits['test'] = 1.
            for split_name in SPLIT_NAMES:
                split_fraction = create_splits[split_name]
                if not 0 <= split_fraction <= 1:  # check split values
                    raise ValueError('incorrect split sizes')
                split_index = int((entries - previous_split_index)
                                  * split_fraction + previous_split_index)
                split = text[previous_split_index:split_index]
                write_txt('{}{}.{}'.format(
                    dataset_dir, split_name, text_name), split)
                previous_split_index = split_index
            assert previous_split_index == entries, f'{previous_split_index} != {entries}'

    # tokenizing
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    max_token_size = tokenizer.max_model_input_sizes[model_name]
    if filtering:
        filtered = '_filtered'
    else:
        filtered = ''
    tensor_dir = f'{dataset_dir}{tokenizer_name}{filtered}/'
    check_make_dir(tensor_dir, True)
    for split_name in splits2tokenize:
        source = read_single_txt('{}{}.{}'.format(
            dataset_dir, split_name, 'source'))
        target = read_single_txt('{}{}.{}'.format(
            dataset_dir, split_name, 'target'))
        text_length = len(source)
        assert text_length == len(target)
        if size:  # optional limitation of samples for tokenization
            source = source[:size]
            target = target[:size]
        log(f'tokenizing target batch for {split_name} of {text_length} samples')
        if filtering:
            target_tokens = tokenizer(target, padding=True)
        else:
            target_tokens = tokenizer(
                target, padding=True, return_tensors="pt")
        if len(target_tokens['attention_mask'][0]) > max_token_size:
            target_tokens = len(target_tokens['attention_mask'][0])
            raise IOError(
                f'target contains more than {max_token_size} tokens: {target_tokens}')
        log(f'tokenizing source batch for {split_name}')
        if filtering:
            source_tokens = tokenizer(
                source, padding='max_length', truncation=True, max_length=max_token_size + 1)
        else:
            source_tokens = tokenizer(
                source, padding='max_length', truncation=True, return_tensors='pt')
        if filtering:  # finding tokenizations that are too long
            tokens_deletes = []
            for i, attention in enumerate(source_tokens['attention_mask']):
                if len(attention) < max_token_size:
                    break
                if attention[max_token_size]:
                    tokens_deletes.append(i)
            deleted_samples = len(tokens_deletes)
            log('{} ({}%) of samples were filtered because they were too long'.format(
                deleted_samples,
                round((deleted_samples / len(source_tokens['attention_mask'])) * 100, 2))
              )
        for text_name, tokens in [('source', source_tokens), ('target', target_tokens)]:
            # creating filtered PyTorch tensors from
            # tokenization lists and replacing them
            if filtering:
                for key in tokens:  # tokens contains `inputs_ids` and `attention_mask`
                    tokens_list = tokens[key]
                    for i in sorted(tokens_deletes, reverse=True):  # actual filtering
                        del tokens_list[i]
                    tokens_tensor = torch.LongTensor(
                        np.array(tokens_list)[:, :512])
                    tokens[key] = tokens_tensor
            tensor_path = f'{tensor_dir}{split_name}_{text_name}.pt'
            log(f'{tensor_path} with output size:',
                tokens[list(tokens.keys())[0]].size())
            assertFileInxestent(tensor_path)
            torch.save(tokens, tensor_path)


if __name__ == "__main__":
    fire.Fire(provide_data)
