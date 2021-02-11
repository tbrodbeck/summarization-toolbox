# Summarization Toolbox

## Introduction
This repository provides an end-to-end pipeline to fine-tune a 🤗-Summary-Model on your own corpus.<br>
It is subdividet in those three parts:
- [Data Provider](#data-provider): Preprocess and Tokenize data for training
- [Model Trainer](#model-trainer): Fine tune a selected 🤗-Model on the provided data
- [Evaluator](#evaluator): Automated evaluation of the fine tuned model on validation set

The pipeline supports _german_ and _english_ texts to be summarized. For both languages a __T5__ model
is used which can further be explored in the paper
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).
<br>  
Huggingface Models:
- german: [t5-base-multi-de-wiki-news](https://huggingface.co/WikinewsSum/t5-base-multi-de-wiki-news)
- english: [t5-base](https://huggingface.co/t5-base)

#### Process Description
1. Provide _source_ and _target_ files with matching text-summary-pairs. These are split and converted to the right format for the fine-tuning task.
2. Choose one of the supported languages and set training parameters via a config file. Then run the training on your data.
3. Evaluate the produced model checkpoints and compare them either by the `Rouge-L` or the specially developed `SemanticSimilarity` metric. You can also track the training metrics via [TensorBoard](https://www.tensorflow.org/tensorboard).

#### Example
The following example was produced by one of our german models which was fine-tuned on our specially scraped [Golem](https://www.golem.de/) corpus. <br>

_Original Text:_<br>
Tamrons neues Objektiv ist ein Weitwinkelzoom für Canon- und Nikonkameras mit Kleinbildsensor, das über 15 Elemente verfügt, darunter dispersionsarme und asphärische. Der sogenannte Silent Drive Motor ermöglicht laut Hersteller eine hohe Geschwindigkeit beim Scharfstellen und eine niedrige Geräuschentwicklung. Die minimale Fokusdistanz wird mit 28 cm angegeben. Die feuchtigkeitsbeständige Konstruktion und die Fluorbeschichtung des Frontelements sollen dazu beitragen, dass das Objektiv auch bei harschen Wetterbedingungen funktioniert. Das Objektiv misst 84 mm x 93 mm und weist einen Filterdurchmesser von 77 mm auf. Das 17-35 mm F2.8-4 Di OSD von Tamron soll Anfang September 2018 für Nikon FX erhältlich sein, ein Canon-EF-Modell wird später folgen. Der Preis wird mit rund 600 US-Dollar angegeben. Deutsche Daten liegen noch nicht vor.

_Produced Summary:_<br>
Tamron hat mit dem 17-35 mm F2.8-4 Di OSD ein Weitwinkelzoom für Canon- und Nikon-Kameras vorgestellt, das über 15 Elemente verfügt.

## Installation

```sh
./install_dependencies.sh
```

## Data Provider

Provides tokenized data for training.

### Input

It requires to have a dataset in the `dataProvider/datasets/$DATASETNAME` directory. Either the dataset can be provided as single files or already split into a train, validation and test set. Each line in a file should represent a single example string.

#### Providing a Single Dataset

The sources (full texts) for summarization should be provided in a `sources.txt` file and the target summarizations should be provided in a `targets.txt` file.

Now the `--create_splits` flag has to be used to create the `train`, `val` and `test` files in that directory, which will then be the resource for the tokenization.

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
bin/provide_data $DATASETNAME $TOKENIZERNAME $MODELNAME <flags>
```

Example:
```sh
bin/provide_data golem WikinewsSum/t5-base-multi-de-wiki-news t5-base --create_splits=True --filtering=True
```

#### Flags

##### `--size=$SIZE`

Limits the amount of samples that are taken for tokenization for each split. Defaults to `None`.

##### `--create_splits=$CREATESPLITS`

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
Additionally you can choose a supported 🤗-Model with the `$MODELNAME` parameter (the model will be downloaded to your virtual environment if you run the training for the first time).<br>  

#### Flags

##### `--filtered=$FILTERED`
By the `$FILTERED` flag you can specify if filtered or unfiltered data is used for training (if previously created by the Data Provider). It defaults to `True`.

##### `--config_name=$CONFIGNAME`
Since all model and training pipeline configurations are read from a config file (which has to be stored in the _./modelTrainer/config_ directory) you might also select your config file by setting the `$CONFIGNAME` parameter.  
If you don't do so, this parameter defaults to _'fine_tuning.ini'_ (which could also be used as a template for your own configurations).

### Usage

Use the Command Line Interface like this:

```sh
bin/run_training $DATASETNAME $MODELNAME <flags>
```

Example:
```sh
bin/run_training golem WikinewsSum/t5-base-multi-de-wiki-news --filtered=False
```

### Configurations

The pipeline is designed to inherit all customizable parameters from an _'.ini'_ file.
It follows the structure that a component is defined by `[COMPONENT]` and the assigned parameters by `parameter = parameter_value` (as string). There are two components, the model and training component. Each component can be configured with with multiple parameters (see fine_tuning.ini for a full list).
Only the parameters in the provided _'fine_tuning_config.ini'_ file stored in the _config_ folders can be changed.

### Output

In the config file you choose an _output_directory_ in this directory the following folder structure is created:
```
output_directory
    └── model_shortname
        └── model_version
            └── checkpoint_folder
            └── final_model_files
        └── logs
            └── model_version
                └── tensorboard_file
```
_<model_shortname>_ = Abbreviation for the chosen model  
_<model_version>_ = Counts the versions of the fine tuned model (canbe seen as an id and makes sure you don't override any previously trained model)  
_<checkpoint_folder>_ = contains model files after a certain number of training steps (checkpoints are saved after n training steps) 
_<tensorboard_file>_ = saved training metrics for TensorBoard usage

After the training the following final output files are saved in the _<model_version>_ folder:
- _config.json_
- _training_args.bin_ (parameters for the [🤗-Trainer](https://huggingface.co/transformers/main_classes/trainer.html))
- _pytorch_model.bin_ (model which can then be loaded for inference)
- _model_info.yml_ (file with information used for evaluation)

## Evaluator

Performs evaluation on the validation or test set for a fine-tuned model.

### Input

To execute the __Evaluation__ you need to previously run the __Model Trainer__ module to generate a fine-tuned  🤗-Model in the right format and stored in the correct folder structure.
These four files are required:
- _config.json_
- _pytorch_model.bin_
- _training_args.bin_
- _model_info.yml_

Since the model evaluation uses the __validation set__ or __test set__ created from the underlying datasaet you need to specify the `$DATASETNAME`.  
Additionally you can choose the fine-tuned 🤗-Model Checkpoints to compare by setting the `$RUNPATH` parameter. This path has to be the directory of the `checkpoint_folder` defined by the folder structure in the training section.

#### Flags

##### `--split_name=$SPLIT_NAME`
Should be `train`, `val` or `test`. Default to __val__
##### `--nr_samples=$NR_SAMPLES`
Number of samples selected from the data set to evaluate the checkpoint on. Defaults to __10__

##### `--metric_type=$METRIC_TYPE`
It can be chosen from the two metric types:
- Rouge-L: set parameter to "Rouge"
- Semantic Similarity: set parameter to "SemanticSimilarity"

Defaults to __Rouge__.

### Usage

Use the Command Line Interface like this:

```sh
bin/evaluate_with_checkpoints_and_compare $RUN_PATH $DATASET_NAME <flags>
```

Example:
```sh
bin/evaluate_with_checkpoints_and_compare golem WikinewsSum/t5-base-multi-de-wiki-news --split_name=train --nr_samples=1000 --metric_type=SemanticSimilarity
```

### Output

By defalut the produced _Overview.xlsx_ files are stored in the __evaluator__ directory under the following structure:
```
evaluator
    └── evaluations
        └── model_short_name
            └── model_version
                └── checkpoint_folders
                    └── metric_type-sample_name-split_name-folders
                        └── iteration-folders
                            - Overview.xlsx
                        - analysis.xlsx
```

## TensorBoard
During the training a `TensorBoard` file is produced which can then be activated to track your training parameters and metrics afterwards in your localhost.
To access the TensorBoard the library _tensorboard_ has to be installed (requirements.txt) and you can use the following CLI to activate it:

```sh
tensorboard --logdir <tensorboard_log_dir>
```
In the <tensorboard_log_dir> a _events.out.tfevents..._-file should exist. The default path is described by the folder structure in the training section.

Example:
```sh
tensorboard --logdir ./results/t5-de/logs/0
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
