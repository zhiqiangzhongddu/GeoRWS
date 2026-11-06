import hashlib
import os
import tempfile
from pathlib import Path

import torch
import yaml

from src.georws.model import GeoRWSModel
from src.georws.perturbations import perturb_affinity
from src.georws.synthesis import synthesize_weights
from src.georws.trainer import train_georws
from src.utils.naming import llm_tag
from src.zsl.model_bases import LINEAR
from src.zsl.regressor import REGRESSOR
from src.zsl.utils import map_label, use_cuda


class GeoRWSRunner(REGRESSOR):
    def __init__(self, _train_x, _train_y, data_loader, _num_class, seed,
                 _cuda=True, train_base=False, _lr=0.001, _beta1=0.5,
                 _num_epoch=20, _batch_size=100, _embed_dim=1000,
                 _num_layers=3, cfg=None, target_weights=None):
        super().__init__(_train_x, _train_y, data_loader, _num_class, seed, cfg,
                         _cuda, train_base, _lr, _beta1, _num_epoch, _batch_size,
                         _embed_dim, _num_layers, target_weights=target_weights)
        self.unseen_model = LINEAR(self.input_dim, len(self.unseen_classes))
        self.ext_model = LINEAR(self.input_dim, len(self.seen_classes) + len(self.unseen_classes))
        if self.cuda:
            self.unseen_model.cuda()
            self.ext_model.cuda()
        self.ext_model.fc.weight.data[:len(self.seen_classes)] = self.target_weights[:, :-1]
        self.ext_model.fc.bias.data[:len(self.seen_classes)] = self.target_weights[:, -1]

    def inject(self, weights, seen_logit_bias):
        weights = weights.to(self.target_weights.device)
        s, d = len(self.seen_classes), self.input_dim
        self.unseen_model.fc.weight.data.copy_(weights[:, :d])
        self.unseen_model.fc.bias.data.copy_(weights[:, d])
        self.ext_model.fc.weight.data[s:].copy_(weights[:, :d])
        self.ext_model.fc.bias.data[s:].copy_(weights[:, d])
        self.ext_model.fc.bias.data[:s].copy_(self.target_weights[:, d] - seen_logit_bias)


def fit_and_evaluate(cfg, data_view, seed, target_weights=None):
    cfg = cfg.clone()
    cfg.zsl.norm_scale_heuristic = cfg.georws.unseen_scale == 'icis'
    runner = GeoRWSRunner(_train_x=data_view.train_feature,
                          _train_y=map_label(data_view.train_label, data_view.seen_classes),
                          data_loader=data_view, _num_class=data_view.num_class, seed=seed,
                          _cuda=use_cuda(cfg), _lr=cfg.train.lr, _beta1=cfg.train.beta1,
                          _num_epoch=cfg.train.num_epoch, _batch_size=cfg.train.batch_size,
                          _embed_dim=cfg.train.embed_dim, _num_layers=cfg.train.num_layers,
                          cfg=cfg, target_weights=target_weights)
    device = runner.target_weights.device
    source = cfg.georws.semantic_source
    base = Path(cfg.text_encoder.cache_dir) / cfg.dataset / cfg.text_encoder.name
    directory = base / llm_tag(cfg)
    if source not in ('pairwise', 'classwise', 'label'):
        raise ValueError(f'Unknown semantic source: {source}')
    path = base / 'label.pt' if source == 'label' else directory / f'{source}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    classes, emb = saved['classes'], saved['emb']
    if classes != list(data_view.class_names) or emb.shape[0] != len(classes):
        raise ValueError(f'Embedding classes mismatch: {path}')
    seen, unseen = data_view.seen_classes, data_view.unseen_classes
    if source == 'pairwise':
        a_seen = emb[seen][:, seen]
        if cfg.georws.perturb.kind == 'text_truncation':
            trunc = directory / f'pairwise_trunc{cfg.georws.perturb.truncation_frac:g}.pt'
            saved_trunc = torch.load(trunc, map_location='cpu', weights_only=True)
            unseen_emb = saved_trunc['emb']
            if (saved_trunc['classes'] != classes or unseen_emb.shape[0] != len(classes)
                    or unseen_emb.shape != emb.shape):
                raise ValueError(f'Truncated embedding mismatch: {trunc}')
        else:
            unseen_emb = emb
        a_unseen = unseen_emb[unseen][:, seen]
    else:
        if cfg.georws.perturb.kind == 'text_truncation':
            raise ValueError('Text truncation requires pairwise semantic source')
        a_seen = emb[seen][:, None, :].expand(-1, len(seen), -1)
        a_unseen = emb[unseen][:, None, :].expand(-1, len(seen), -1)
    kind = cfg.georws.perturb.kind
    if kind not in ('none', 'text_truncation'):
        generator = torch.Generator().manual_seed(cfg.georws.perturb.seed + seed)
        a_unseen = perturb_affinity(a_unseen, kind, cfg.georws.perturb.rate,
                                    cfg.georws.perturb.sigma, generator)
    model = GeoRWSModel(a_seen.size(-1), runner.target_weights.size(-1),
                        cfg.georws.latent_dim, cfg.georws.hidden_dim,
                        cfg.georws.adaptive_encoder).to(device)

    def synthesize():
        return synthesize_weights(model, a_unseen, runner.target_weights,
                                  cfg.georws.infer_batch_size, cfg.georws.unseen_scale, device)

    def evaluate(epoch):
        weights, _ = synthesize()
        runner.inject(weights, cfg.zsl.seen_logit_bias)
        metrics = runner.val_gzsl()
        print(f'GeoRWS diagnostic epoch={epoch}: {tuple(float(x) for x in metrics)}')

    train_georws(model, a_seen, runner.target_weights, cfg, device, 'GeoRWS',
                 eval_fn=evaluate if cfg.zsl.eval_every > 0 else None)
    weights, alpha = synthesize()
    tag = f'{llm_tag(cfg)}_{cfg.text_encoder.name}_{source}'
    if not cfg.georws.use_geo:
        tag += '_nogeo'
    if not cfg.georws.use_cons:
        tag += '_nocons'
    if not cfg.georws.adaptive_encoder:
        tag += '_static'
    if kind != 'none':
        tag += f'_{kind}'
    fingerprint = yaml.safe_dump(dict(georws=yaml.safe_load(cfg.georws.dump()),
                                      seen_logit_bias=cfg.zsl.seen_logit_bias,
                                      train=yaml.safe_load(cfg.train.dump()),
                                      image_embedding=cfg.zsl.image_embedding), sort_keys=True)
    tag += '_' + hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:8]
    path = Path(cfg.output_root) / 'georws' / cfg.dataset / tag / f'{cfg.zsl.mode}_seed{seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(dict(alpha=alpha, weights=weights, unseen_classes=unseen.cpu(),
                        seen_classes=seen.cpu(), cfg=cfg.dump()), tmp)
    os.replace(tmp.name, path)
    runner.inject(weights, cfg.zsl.seen_logit_bias)
    _, acc_seen, acc_unseen, h, acc_zsl = runner.val_gzsl()
    return dict(acc_zsl=float(acc_zsl), acc_unseen=float(acc_unseen), acc_seen=float(acc_seen), H=float(h))


def run_georws(cfg, data, seed):
    if cfg.zsl.mode == 'validate':
        from src.georws.validation import validate_georws
        return validate_georws(cfg, data, seed)
    return fit_and_evaluate(cfg, data, seed)
