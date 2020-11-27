import sys
sys.path.append(".")
import fire
import os
import torch
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from utilities.gerneral_io_utils import check_make_dir, read_single_txt
from evaluator.metrics import SemanticSimilarityMetric
from evaluator.eval import run_evaluation
from timelogging.timeLog import log

DATA_DIR = "./dataProvider/datasets"

def evaluate(data_set_name: str, modelDir: str, model_name: str, evaluation_parameters):
    """
    run evaluation
    :param reference_to_base: 
    :param data_set_name:
    :param model_name:
    :param config_path:
    :return:
    """

    print("\n")
    log("Received parameters for evaluation:")
    for p in evaluation_parameters:
        log(f"- {p}: {evaluation_parameters[p]}")

    ###################################
    # Initialize model
    ###################################
    model = AbstractiveSummarizer(
        evaluation_parameters["language"],
        modelDir
    )

    # initialize reference model

    if evaluation_parameters["reference_model"] == "True":
        reference_model = AbstractiveSummarizer(
            evaluation_parameters["language"],
            "base"
        )
    else:
        reference_model = None

    # check if output directory exists
    out_dir = evaluation_parameters['output_directory']
    if not check_make_dir(out_dir, create_dir=True):
        log(f"Created output directory'{out_dir}'")

    samples = int(evaluation_parameters['number_samples'])

    ###################################
    # Load evaluation data
    ###################################
    data_set_dir = os.path.join(DATA_DIR, data_set_name)
    assert check_make_dir(data_set_dir), f"Data set '{data_set_name}' not directory '{DATA_DIR}'. " \
                                         f"Please store data there!"

    model_folder = model_name.split("/")[0]
    tensor_folder = model_name.split("/")[1]
    tensor_dir = os.path.join(
        data_set_dir,
        model_folder,
        tensor_folder
    )
    try:
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir)
    except:
        tensor_dir += "_filtered"
        assert check_make_dir(tensor_dir) and os.listdir(tensor_dir), \
            f"Neither '{tensor_dir.rstrip('_filtered')}' not '{tensor_dir}' does exist or it is empty!"

    source_path = os.path.join(tensor_dir, "test_source.pt")
    target_path = os.path.join(tensor_dir, "test_target.pt")
    assert os.path.isfile(source_path) and os.path.isfile(target_path), \
        f"Data pair '{source_path}' and '{target_path}' does not exist!"

    evaluation_dict = {
        "source": torch.load(open(source_path, "rb")),
        "target": torch.load(open(target_path, "rb"))
    }

    metric = None
    if evaluation_parameters["metric"] == "SemanticSimilarity":
        metric = SemanticSimilarityMetric(evaluation_parameters["language"])

    run_evaluation(evaluation_dict, model, metric, out_dir, samples, reference_model)

def predict(modelPaths: List):
    for modelPath in modelPaths:
        pass

def newEval(runPath: str, tokenizerName: str, datasetName: str, language: str, ):
    evaluation_parameters = {
        "language": "german",
        "checkpointEvaluation": False,
        "output_directory": "evaluator/output",
        "number_samples": 5,
        "reference_model": True,
    }
    dataDir = f'dataProvider/datasets/{datasetName}/'
    sourceText = read_single_txt('{}{}.{}'.format(dataDir, 'val', 'source'))
    targetText = read_single_txt('{}{}.{}'.format(dataDir, 'val', 'target'))
    walk = os.walk(runPath)
    try:  # collects checkpoints if the run contains checkpoints
        _, models, _ = next(walk)
        log('Checkpoints:', models)
        modelPaths = []
        for checkpoint in models:
            modelPaths.append(runPath + '/' + checkpoint)
            predict(modelPaths)
    except StopIteration:  # else just take the run path as model path
        log('no checkpoints')
        modelPaths = [runPath]
    predict(modelPaths)



if __name__ == '__main__':
  fire.Fire(evaluate)
