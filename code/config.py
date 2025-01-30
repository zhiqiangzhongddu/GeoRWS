import os
import argparse
from pathlib import PurePath
from yacs.config import CfgNode as CN

from code.utility.utils import project_root_path


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Parameters for image tasks
    cfg.cv = CN()
    # Parameters for nlp tasks
    cfg.nlp = CN()
    # Parameters for graph tasks
    cfg.graph = CN()
    # Parameters to query LLM
    cfg.llm = CN()

    # ------------------------------------------------------------------------ #
    # General options
    # ------------------------------------------------------------------------ #
    # Whether to use Demo Test mode
    cfg.demo_test = False
    # Number of samples for demo test
    cfg.num_sample = 10
    # Fix the running seed to remove randomness
    cfg.seed = 42
    # Number of runs
    cfg.num_runs = 1
    # Whether perform experiment of model transfer from one dataset to another
    cfg.zst = False
    # Transfer from which dataset: imagenet, cub, sun, awa2
    cfg.zst_from = "imagenet"

    # ------------------------------------------------------------------------ #
    # Data options
    # ------------------------------------------------------------------------ #
    # Dataset
    cfg.dataset = "CUB"
    # # Whether data in matlab format
    # cfg.matdataset = True

    # ------------------------------------------------------------------------ #
    # Output options
    # ------------------------------------------------------------------------ #
    # Whether save matrices with predictions after evaluation
    cfg.save_pred_matrix = False
    # Whether only validate after final epoch, when running on test set,
    cfg.strict_eval = False
    # Whether enable early stopping heuristic
    cfg.early_stopping_slope = False
    # Whether enable cosine similarity loss
    cfg.cos_sim_loss = False
    # Whether to include unseen attributes during training
    cfg.include_unseen = False
    # Whether scale the predicted classifier weights (heuristic for bias correction)
    cfg.norm_scale_heuristic = False

    # ------------------------------------------------------------------------ #
    # LLM Model options
    # ------------------------------------------------------------------------ #
    # LLM provider options
    cfg.llm.provider = "openai"
    # LLM model name
    # gpt-4o, gpt-4-turbo
    cfg.llm.name = "gpt-4o"
    cfg.llm.temperature = 1.
    cfg.llm.top_p = 1.
    cfg.llm.frequency_penalty = 0.
    cfg.llm.presence_penalty = 0.
    # temperature: Defaults to 1 (suggest 0.6?)
    #               What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
    #               make the output more random, while lower values like 0.2 will make it more
    #               focused and deterministic.
    # top_p: Defaults to 1 (suggest 0.9?)
    #               An alternative to sampling with temperature, called nucleus sampling, where the
    #               model considers the results of the tokens with top_p probability mass. So 0.1
    #               means only the tokens comprising the top 10% probability mass are considered.
    #               We generally recommend altering this or `top_p` but not both.
    # frequency_penalty: Defaults to 0
    #               Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    #               existing frequency in the text so far, decreasing the model's likelihood to
    #               repeat the same line verbatim.
    # presence_penalty: Defaults to 0
    #               Number between -2.0 and 2.0. Positive values penalize new tokens based on
    #               whether they appear in the text so far, increasing the model's likelihood to
    #               talk about new topics.
    #               [See more information about frequency and presence penalties.]
    #               (https://platform.openai.com/docs/guides/text-generation/parameter-details)

    # ------------------------------------------------------------------------ #
    # CV Task options
    # ------------------------------------------------------------------------ #
    # The base classifier: res101_finetuned, pretrained_res101
    cfg.cv.image_embedding = "res101_finetuned"
    # Semantic class-wise information
    cfg.cv.class_embedding = "att"
    # Whether run ConSE benchmark
    cfg.cv.conse_benchmark = False
    # Whether run COSTA benchmark
    cfg.cv.costa_benchmark = False
    # Adapted baseline from Akyürek et al.
    # Project predicted weights unto subspace spanned by seen class weights
    cfg.cv.subspace_proj = False
    # Run VGSE CRM baseline (choices: wavg or smo)
    cfg.cv.vgse_baseline = None
    # Number of VGSE CRM WAvg neighbours
    cfg.cv.vgse_nbs = 5
    # eta hyperparameter for VGSE CRM WAvg
    cfg.cv.vgse_eta = 5
    # alpha hyperparameter for VGSE CRM SMO
    cfg.cv.vgse_alpha = 0.
    # Run wDAE-GNN benchmark
    cfg.cv.daegnn = False
    # Train a single autoencoder predicting weights from attributes
    cfg.cv.single_autoencoder_baseline = False
    # Ablation: remove Weight to Attribute mapping
    cfg.cv.single_modal_ablation = False
    # Run ablation with reducing number of seen classes (0 = No ablation)
    cfg.cv.class_reduction_ablation = 0
    # Calculate output distribution on test set of seen and unseen classes
    cfg.cv.calc_entropy = False

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    # Method: icis, conse, costa, subreg, wdae, wavg, smo
    cfg.method = "icis"
    # Learning rate
    cfg.lr = 0.0001
    # Max number of epochs to train
    cfg.num_epoch = 1000
    # Input batch size
    cfg.batch_size = 16
    # Dimensionality of the hidden layers
    cfg.embed_dim = 1000
    # Number of layers in weight prediction MLP: 2, 3, 4
    cfg.num_layers = 2
    # beta1 parameter(s) for adam to train weight regressor network.
    cfg.beta1 = 0.5
    # Max number of epochs to train classifier
    cfg.classifier_num_epoch = 100
    # Learning rate to train softmax classifier
    cfg.classifier_lr = 0.0001
    # beta1 for adam to train classifier. default=0.5
    cfg.classifier_beta1 = 0.5

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    # Update from task-specific information
    # Add task type: cv, nlp, graph
    if cfg.dataset in ["AWA2", "CUB", "SUN"]:
        cfg.task_type = "cv"
    else:
        raise ValueError("Unrecognized dataset {}".format(cfg.dataset))
    # Path to datasets folder
    cfg.data_root = str(PurePath(
        project_root_path,
        "data", cfg.task_type,
    ))
    # Path to input folder
    cfg.input_root = str(PurePath(
        project_root_path,
        "input", cfg.task_type,
    ))
    # Path to save model checkpoints and results
    cfg.out_root = str(PurePath(
        project_root_path,
        "output", cfg.task_type,
    ))

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
