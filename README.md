# Summarization Toolbox

## Installation

```sh
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

If training, validation and test splits are already present, they should be provided in the following format of [🤗-seq2seq examples](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).

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

```sh
python dataProvider/main.py $DATASETNAME $TOKENIZERNAME $MODELNAME <flags>
```

#### Flags

##### `--size=$SIZE`

Limits the amount of samples that are taken for tokenization for each split. Defaults to `None`.

##### `--createSplits=$CREATESPLITS`

Split the dataset into train, validation and test splits. Defaults to `False`.

`$CREATESPLITS` has to be a dictionary containing the keys `train` and `val` and values between 0 and 1. The value of `train` represents the ratio of the dataset that is used for training (and not for validation or testing). The value of `val` represents the the ratio between the validation and the test set. Because of shell restrictions the dictionary has to be wrapped in `"` in the CLI, like this: `--createSplits="{'train': 0.7, 'val': 0.66}"`

If the value of `$CREATESPLITS` is `True` it defaults to `{'train': 0.8, 'val': 0.5}`, which results a 80/10/10 split.

##### `--splits2tokenize=$SPLITS2TOKENIZE`

Can be set to only tokenize certain splits. Defaults to `[train, val, test]`.

##### `--filtering=$FILTERING`

Longer examples than the maximum token size are filtered, else they are truncated. Defaults to `True`.

### Output

The resulting tokenized [PyTorch](https://pytorch.org/) tensors are saved in the `dataProvider/datasets/$DATASETNAME/$TOKENIZERNAME[_filtered]` directory as the following files:

```
train_source.pt
train_target.pt
val_source.pt
val_target.pt
test_source.pt
test_target.pt
```

## Model Trainer

Performs training process for selected model on the previously created data sets.

### Input

To execute the __Model Training__ you need to previously run the __Data Provider__ module to generate training data in the right format either from your own or predefined text/summary pairs.
It requires files in the output format of the __Data Provider__ module. Since you could have run the module for multiple text/summary sets, you have to provide the `$DATASETNAME` to train on.  
Additionally you can choose a supported 🤗-Model with the `$MODELNAME` parameter (the model will be downloaded to your virtual environment if you run the training for the first time).
Since all model and training pipeline configurations are read from a config file (which has to be stored in the _./config_ directory) you might also select your config file by setting the `$CONFIGNAME` parameter.  
If you don't do so, this parameter defaults to _'fine_tuning.ini'_ (which could also be used as a template for your own configurations).

### Usage

Use the Command Line Interface like this:

```sh
python modelTrainer/main.py $DATASETNAME $MODELNAME $CONFIGNAME
```

### Configurations

The pipeline is designed to inherit all customizable parameters from an _'.ini'_ file.
It follows the structure that a component is defined by `[COMPONENT]` and the assigned parameters by `parameter = parameter_value` (as string).
Only the parameters in the provided _'fine_tuning_config.ini'_ file stored in the _config_ folders can be changed.

### Output

In the config file you choose an _output_directory_ in this directory the following folder structure is created:
```
output_directory
    └── logs
    └── <model_shortname>
        └── <model_version>
            └── <checkpoint_folder>
```
_<model_shortname>_ = Abbreviation for the chosen model  
_<model_version>_ = Counts the versions (no override)  
_<checkpoint_folder>_ = states of the model after a certain number of training steps

After the training the following final output files are saved in the _<model_version>_ folder:
- _config.json_
- _training_args.bin_ (parameters for the [🤗-Trainer](https://huggingface.co/transformers/main_classes/trainer.html))
- _pytorch_model.bin_ (model which can then be loaded for inference)

## Evaluator
Example command:
```sh
python evaluator/main.py modelTrainer/results/t5-de/0/config.json WikinewsSum/t5-base-multi-de-wiki-news t5-base golem
```

## Development Instructions

```sh
pip install pytest
```

Use [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```sh
fd . | entr pytest
```

Use the following command to add a new package (optionally with version number) `$pkg` to the repository, while keeping `requirements.txt` orderly:

```sh
echo $pkg | sort -o requirements.txt - requirements.txt && pip install $pkg
```
