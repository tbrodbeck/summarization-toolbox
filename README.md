# Summarization Toolbox

## Installation

```
pip install -r requirements.txt
```

## Data Provider

Call it with `python -m dataProvider.main` like this:

SYNOPSIS
`python -m dataProvider.main DATASETNAME TOKENIZERNAME <flags>`

DESCRIPTION
Provides tokenized data for training

POSITIONAL ARGUMENTS
`DATASETNAME`
`TOKENIZERNAME`

FLAGS
`--size=SIZE`
Defaults to None.
`--createSplits=CREATESPLITS`
Split the dataset into train, validation and test splits. Defaults to None.
`--splits2tokenize=SPLITS2TOKENIZE`
Can be set to only tokenize certain splits. Defaults to SPLIT_NAMES.

NOTES
You can also use flags syntax for POSITIONAL ARGUMENTS

## Development Instructions

```
pip install pytest
```

Use [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```
fd . | entr python -m pytest
```
