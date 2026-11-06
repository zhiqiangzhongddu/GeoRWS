import os
import argparse

from yacs.config import CfgNode as CN

from ._common import set_general_cfg
from ._zsl import set_zsl_cfg
from ._base_classifier import set_base_classifier_cfg
from ._train import set_train_cfg
from ._baselines import set_baselines_cfg
from ._llm import set_llm_cfg
from ._descriptions import set_descriptions_cfg
from ._text_encoder import set_text_encoder_cfg
from ._georws import set_georws_cfg
from ._save_results import set_save_results_cfg


def set_cfg(cfg: CN) -> CN:
    set_general_cfg(cfg)
    set_zsl_cfg(cfg)
    set_base_classifier_cfg(cfg)
    set_train_cfg(cfg)
    set_baselines_cfg(cfg)
    set_llm_cfg(cfg)
    set_descriptions_cfg(cfg)
    set_text_encoder_cfg(cfg)
    set_georws_cfg(cfg)
    set_save_results_cfg(cfg)
    return cfg


def update_cfg(cfg: CN, argv: list[str] | None = None) -> CN:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", metavar="FILE", help="Path to config file")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args(argv)
    cfg = cfg.clone()
    if args.config:
        if not os.path.isfile(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    return cfg


cfg = set_cfg(cfg=CN())
