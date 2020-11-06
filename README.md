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

#### Flags

##### `--size=$SIZE`

Defaults to `None`.

##### `--createSplits=$CREATESPLITS`

Split the dataset into train, validation and test splits. Defaults to `None`.

`$CREATESPLITS` has to be a dictionary containing the keys `train` and `val` and values between 0 and 1. The value of `train` represents the ratio of the dataset that is used for training (and not for validation or testing). The value of `val` represents the the ratio between the validation and the test set.

If the value of `$CREATESPLITS` is `True` it defaults to `{'train': 0.8, 'val': 0.5}`, which results a 80/10/10 split.

##### `--splits2tokenize=$SPLITS2TOKENIZE`

Can be set to only tokenize certain splits. Defaults to `[train, val, test]`.

### Output

The resulting tokenized [PyTorch](https://pytorch.org/) tensors are saved in the `dataProvider/datasets/$DATASETNAME/tensors` directory as the following files:

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

To execute the `Model Training` you need to previously run the `Data Provider` module to generate training data in the right format either from your own or predefined text/summary pairs.
It requires files in the output format of the `Data Provider` module. Since you could have run the module for multiple text/summary sets, you have to provide the `$DATASETNAME` to train on.  
Additionally you can choose a supported 🤗-model with the `$MODELNAME` parameter (the model will be downloaded to your virtual environment if you run the training for the first time).
Since all model and training pipeline configurations are read from a config file (which has to be stored in the _./config_ directory) you might also select your config file by setting the `$CONFINAME` parameter.  
If you don't do so, this parameter defaults to _'fine_tuning.ini'_ (which could also be used as a template for your own configurations).

### Usage

Use the Command Line Interface like this:

```bash
python -m modelTrainer.main $DATASETNAME $MODELNAME $CONFINAME
```

### Configurations

The pipeline is designed to inherit all customizable parameters from an _'.ini'_ file.
It fallows the structure that a component is defined by `\[COMPONENT]` and the assigned parameters by _parameter = parameter_value_ (as string).
Only the parameters in the provided _'fine_tuning_config.ini'_ file stored in the _config_ folders can be changed.

### Output

In config file you choose an _output_directory_ in this directory the following folder structur is created:
```
output_directory
    └── logs
    └── <model_shortname>
        └── <model_version>
            └── <checkpoin_files>
```
<model_shortname> = Abbriviation for the chosen model
<model_version> = Counts the versions (no override)
<checkpoint_folders> = states of the model after a certain nuber of training steps

After the training the following final output files are saved in the <model_version> folder:
- _config.json_
- _training_args.pin_ (parameters for the [🤗-Trainer](https://huggingface.co/transformers/main_classes/trainer.html))
- _pytorch_model.bin_ (model which can then be loaded for inference)


## Development Instructions

```
pip install pytest
```

Use [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```
fd . | entr python -m pytest
```
