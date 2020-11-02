# Summarization Toolbox

## Installation

```
pip install -r requirements.txt
```

## Data Provider

Provides tokenized data for training. Use it with like this:

### Synopsis

`python -m dataProvider.main DATASETNAME TOKENIZERNAME <flags>`

### Positional Arguments

`DATASETNAME`
`TOKENIZERNAME`

### Flags

`--size=SIZE`
Defaults to None.

`--createSplits=CREATESPLITS`
Split the dataset into train, validation and test splits. Defaults to None.

`--splits2tokenize=SPLITS2TOKENIZE`
Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.

### Notes

You can also use flags syntax for POSITIONAL ARGUMENTS

## Development Instructions

```
pip install pytest
```

Use [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```
fd . | entr python -m pytest
```
