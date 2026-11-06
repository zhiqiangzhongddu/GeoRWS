import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.config import cfg as default_cfg, update_cfg
from src.data_loader.dataset import ZSLDataset
from src.zsl.base_classifier import BASECLASSIFIER
from src.zsl.cache import base_classifier_path
from src.zsl.utils import map_label, set_seed, use_cuda
from src.utils.save_results import append_result_row


def prepare_base_model(cfg, data, seed, tag='', eval_x=None, eval_y=None):
    path = base_classifier_path(cfg, seed, tag)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # The original used train.batch_size; the dedicated classifier batch size defaults to the same 16.
    model = BASECLASSIFIER(
        _train_x=data.train_feature, _train_y=map_label(data.train_label, data.seen_classes),
        data_loader=data, _num_class=data.num_class, seed=seed, _cuda=use_cuda(cfg),
        _lr=cfg.train.lr, _beta1=cfg.train.beta1, _num_epoch=cfg.base_classifier.num_epoch,
        _batch_size=cfg.base_classifier.batch_size, _embed_dim=cfg.train.embed_dim,
        _num_layers=cfg.train.num_layers, cfg=cfg, eval_x=eval_x, eval_y=eval_y,
    ).fit()
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        torch.save({'state_dict': model.state_dict(), 'input_dim': model.fc.in_features,
                    'num_classes': model.fc.out_features}, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def run_zsl(cfg, overrides=None):
    if cfg.zsl.mode not in ('test', 'validate'):
        raise ValueError(f'Unknown ZSL mode: {cfg.zsl.mode}')
    if cfg.zsl.mode == 'validate' and cfg.method != 'georws':
        raise ValueError('Validation mode is only supported for georws')
    if cfg.num_runs < 1 or cfg.num_runs > len(cfg.seeds):
        raise ValueError('num_runs must be between 1 and the number of seeds')
    if use_cuda(cfg):
        torch.cuda.set_device(cfg.device)
    seeds = cfg.seeds[:cfg.num_runs]
    results = []
    for seed in seeds:
        set_seed(seed)
        data = ZSLDataset(cfg)
        # GeoRWS validation must prepare its own held-out head without reading test images.
        if cfg.zsl.mode == 'test':
            prepare_base_model(cfg, data, seed)
        set_seed(seed)
        if cfg.method == 'georws':
            from src.georws.run import run_georws
            result = run_georws(cfg, data, seed)
        else:
            method_cfg = cfg.clone()
            method_cfg.baselines.subspace_proj = cfg.method == 'subreg'
            method_cfg.baselines.daegnn = cfg.method == 'wdae'
            method_cfg.baselines.single_autoencoder_baseline = cfg.method in ('subreg', 'wdae')
            method_cfg.baselines.vgse_baseline = cfg.method if cfg.method in ('wavg', 'smo') else ''
            kwargs = dict(_train_x=data.train_feature, _train_y=data.train_mapped_label,
                          data_loader=data, _num_class=data.num_class, seed=seed,
                          _cuda=use_cuda(cfg), _lr=cfg.train.lr, _beta1=cfg.train.beta1,
                          _num_epoch=cfg.train.num_epoch, _batch_size=cfg.train.batch_size,
                          _embed_dim=cfg.train.embed_dim, _num_layers=cfg.train.num_layers, cfg=method_cfg)
            if cfg.method == 'conse':
                from src.baselines.conse import ConSE
                model = ConSE(**kwargs)
            elif cfg.method == 'costa':
                from src.baselines.costa import COSTA
                model = COSTA(**kwargs)
            elif cfg.method in ('wavg', 'smo'):
                from src.baselines.vgse import VGSE_CRM
                model = VGSE_CRM(**kwargs)
            elif cfg.method in ('icis', 'subreg', 'wdae'):
                from src.baselines.icis import Joint
                model = Joint(**kwargs)
                model.fit()
            else:
                raise ValueError(f'Unknown method: {cfg.method}')
            result = dict(acc_zsl=model.acc_unseen_zsl, acc_unseen=model.acc_unseen,
                          acc_seen=model.acc_seen, H=model.H)
        result = {k: float(result[k]) for k in ('acc_zsl', 'acc_unseen', 'acc_seen', 'H')}
        results.append(result)
        print(f'Seed {seed}: {result}')
    row = dict(method=cfg.method, dataset=cfg.dataset, mode=cfg.zsl.mode,
               image_embedding=cfg.zsl.image_embedding, class_embedding=cfg.zsl.class_embedding,
               llm=cfg.llm.name, text_encoder=cfg.text_encoder.name,
               semantic_source=cfg.georws.semantic_source, unseen_scale=cfg.georws.unseen_scale,
               seen_logit_bias=cfg.zsl.seen_logit_bias, conse_fixed_eval=cfg.baselines.conse_fixed_eval,
               seeds=','.join(map(str, seeds)), overrides=' '.join(overrides or []))
    if cfg.method == 'georws':
        row.update(perturb_kind=cfg.georws.perturb.kind, perturb_rate=cfg.georws.perturb.rate,
                   perturb_sigma=cfg.georws.perturb.sigma,
                   perturb_truncation_frac=cfg.georws.perturb.truncation_frac,
                   perturb_seed=cfg.georws.perturb.seed)
    for metric in ('acc_zsl', 'acc_unseen', 'acc_seen', 'H'):
        values = [r[metric] for r in results]
        row[f'{metric}_mean'] = float(np.mean(values))
        row[f'{metric}_std'] = float(np.std(values, ddof=0))
        row[f'{metric}_runs'] = ','.join(map(str, values))
        print(f"{metric}: {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
    if cfg.save_results.enabled:
        append_result_row(Path(cfg.save_results.output_dir) / 'zsl.tsv', row)
    return results


def run_zsl_from_cli(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cfg = update_cfg(default_cfg, argv)
    overrides = [f'{key}={value}' for key, value in zip(args.opts[::2], args.opts[1::2])]
    run_zsl(cfg, overrides=overrides)
    return 0
