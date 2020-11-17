"""
script to fine tune a huggingface model
"""

import os
from timelogging.timeLog import log
from modelTrainer.data_set_creation import create_dataset
from transformers import Trainer, TrainingArguments
from utilities.gerneral_io_utils import check_make_dir


def fine_tune_model(summary_model, results_path: str, data_dict: dict, parameters: dict):
    """
    fine tuning pipeline for the summary model
    :param summary_model:
    :param results_path:
    :param data_dict:
    :param parameters:
    :return:
    """

    # get dataset
    if "val" in data_dict:
        train_data, val_data = create_dataset(
            (data_dict["train"]["source"], data_dict["train"]["target"]),
            (data_dict["val"]["source"], data_dict["val"]["target"])
        )
    else:
        train_data, val_data = create_dataset(
            (data_dict["train"]["source"], data_dict["train"]["target"])
        )

    # recursively create output directory
    check_make_dir(results_path)
    model_type = summary_model.short_name
    check_make_dir(results_path + "/" + model_type)
    model_version = 0

    final_path = os.path.join(results_path, model_type, str(model_version))
    while check_make_dir(final_path, create_dir=True):
        if len(os.listdir(final_path)) == 0:
            break
        model_version += 1
        final_path = os.path.join(results_path, model_type, str(model_version))

    # prepare path for logs
    logs_path = os.path.join('/'.join(final_path.split('/')[:-2]), 'logs')
    if not check_make_dir(logs_path, create_dir=True):
        log("Created", logs_path)

    # initialize the training parameters
    training_args = TrainingArguments(
        output_dir=final_path,  # output directory
        num_train_epochs=int(parameters["epochs"]),  # total number of training epochs
        per_device_train_batch_size=int(parameters["train_batch_size"]),  # batch size per device during training
        per_device_eval_batch_size=int(parameters["val_batch_size"]) if val_data else None,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logs_path,  # directory for storing logs
        logging_steps=100,
        save_steps=int(parameters["checkpoint_steps"]),
        do_train=True,
        do_eval=True if val_data else False
    )

    # initialize the trainer class
    trainer = Trainer(
        model=summary_model.model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=val_data if val_data else None,  # evaluation dataset
    )

    # perform the training
    trainer.train()

    # save the fine tuned model
    trainer.save_model()
