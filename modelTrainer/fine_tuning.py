"""
script to fine tune a huggingface model
"""

import os
from modelTrainer.data_set_creation import create_dataset
from transformers import Trainer, TrainingArguments
from utilities.gerneral_io_utils import read_data, check_make_dir


def fine_tune_model(summary_model, input_path, results_path, text_name, summary_name, train_batch_size=8, components = []):
    """
    fine tuning pipeline for the summary model
    :param summary_name:
    :param text_name:
    :param summary_model:
    :param input_path:
    :param results_path:
    :return:
    """
    # set model to training mode
    summary_model.set_mode("train")

    data = read_data(input_path, text_name, summary_name, limit=10000)
    # automated data set creation
    train_data, val_data = create_dataset(data, summary_model)

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
        print("\nCreated", logs_path)

    # save the current training an evaluation set
    #for name, item in [("train_data", train_data), ("val_data", val_data)]:
        #write_pickle(item, name, final_path)

    # initialize the training parameters
    training_args = TrainingArguments(
        output_dir=final_path,  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        #per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logs_path,  # directory for storing logs
        logging_steps=100,
        do_train=True
    )

    # initialize the trainer class
    trainer = Trainer(
        model=summary_model.model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        #eval_dataset=val_data,  # evaluation dataset
    )

    # perform the training
    trainer.train()

    # save the fine tuned model
    trainer.save_model()
