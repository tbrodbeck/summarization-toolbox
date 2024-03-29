"""
script to fine tune a huggingface model
"""

import os
import pickle
from timelogging.timeLogLater import logOnce as log
from transformers import Trainer, TrainingArguments
import yaml
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from modelTrainer.data_set_creation import create_dataset
from utilities.io_utils import check_make_dir
from utilities.cleaning_utils import limit_data


def fine_tune_model(
        summary_model: AbstractiveSummarizer,
        results_path: str,
        data_dict: dict,
        parameters: dict):
    """fine tuning pipeline that runs the training

    Args:
        summary_model (AbstractiveSummarizer): model to train on
        results_path (str): store resulting models and checkpoints
        data_dict (dict): data for training and optional evaluation
        parameters (dict): training parameters
    """

    # limit samples taken into
    # account for training
    data_dict["train"] = limit_data(
        data_dict["train"], int(parameters["number_samples"]))

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

    if parameters["limit_val_data"] != "-1":
        val_data = val_data[:min(
            len(val_data), int(parameters["limit_val_data"])
        )]

    # recursively create output directory
    check_make_dir(results_path)
    model_type = summary_model.short_name
    check_make_dir(results_path + "/" + model_type)
    model_version = 0

    final_path = os.path.join(results_path, model_type, str(model_version))
    logs_path = os.path.join(results_path, model_type, "logs", str(model_version))
    while check_make_dir(final_path, create_dir=True):
        if len(os.listdir(final_path)) == 0:
            break
        model_version += 1
        final_path = os.path.join(results_path, model_type, str(model_version))
        logs_path = os.path.join(results_path, model_type, "logs", str(model_version))

    # prepare path for logs
    check_make_dir(logs_path, create_dir=True)

    # initialize the training parameters
    training_args = TrainingArguments(
        output_dir=final_path,
        num_train_epochs=int(parameters["epochs"]),
        per_device_train_batch_size=int(parameters["train_batch_size"]),
        per_device_eval_batch_size=int(
            parameters["val_batch_size"]
        ) if val_data else None,
        do_eval=bool(val_data),
        eval_steps=int(parameters["eval_steps"]),
        evaluate_during_training=bool(val_data),
        warmup_steps=int(parameters["warmup_steps"]),
        weight_decay=float(parameters["weight_decay"]),
        logging_dir=logs_path,
        logging_steps=int(parameters["logging_steps"]),
        logging_first_step=True,
        save_steps=int(parameters["checkpoint_steps"]),
        do_train=True
    )

    # initialize the trainer class
    trainer = Trainer(
        model=summary_model.model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        prediction_loss_only=True # loss for logging
    )

    log("started training")
    # perform the training
    try:
        training_history = trainer.train()
        # save training history
        with open(final_path + "/training_history.pickle", "wb") as history_file:
            pickle.dump(training_history, history_file)

    finally:
        # save info file
        info_dict = {
            "language": summary_model.language,
            "model_name": summary_model.model_name,
            "run_name": summary_model.short_name + "/" + str(model_version),
            "total_iterations": int(len(train_data) / int(parameters["train_batch_size"]))
                                 * int(parameters["epochs"])
        }

        with open(final_path + "/model_info.yml", "w") as info_file:
            yaml.dump(info_dict, info_file)
        # save the fine tuned model
        check_make_dir(final_path, True)
        trainer.save_model(final_path)
