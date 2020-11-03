# Summarization Toolbox

## Installation

```
pip install -r requirements.txt
```

## Data Provider

Provides tokenized data for training.

### Input

It requires to have a dataset in the `dataProvider/datasets/$DATASETNAME` directory. Either the dataset can be provided as single files or already split into a train, validation and test set. Each line in a file should represent a single example string.

#### Providing a Single Dataset

The sources (full texts) for summarization should be provided in a `sources.txt` file and the target summarizations should be provided in a `targets.txt` file.

Now the `--createSplits` flag has to be used to create the `train`, `val` and `test` files in that directory, which will then be the resource for the tokenization.

#### Providing Train, Val and Test Split Files

If training, validation and test splits are already present, they should be provided in the following format of [🤗](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).

```
train.source
train.target
val.source
val.target
test.source
test.target
```

### Usage

Use the Command Line Interface like this:

```
python -m dataProvider.main $DATASETNAME $TOKENIZERNAME $MODELNAME <flags>
```

#### Positional Arguments

`DATASETNAME`

`TOKENIZERNAME`

`MODELNAME`

#### Flags

`--size=$SIZE`
Defaults to None.

`--createSplits=$CREATESPLITS`
Split the dataset into train, validation and test splits. Defaults to None.

`--splits2tokenize=$SPLITS2TOKENIZE`
Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.

#### Notes

You can also use flags syntax for POSITIONAL ARGUMENTS

### Output

The resulting tokenized [PyTorch](https://pytorch.org/) tensors are saved in the `dataProvider/datasets/$DATASETNAME/tensors` directory.

## Development Instructions

```
pip install pytest
```

Use [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```
fd . | entr python -m pytest
```
