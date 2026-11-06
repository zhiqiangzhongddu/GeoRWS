import hashlib
import json
import math
from pathlib import Path

import torch

from src.descriptions.generate import _names, check_manifest, load_descriptions, required_pair_keys
from src.utils.naming import llm_tag


def truncate_text(text, frac):
    words = text.split()
    return ' '.join(words[:math.ceil(frac * len(words))])


def load_embeddings(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Embedding cache not found: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)


def build_embeddings(cfg, data, encoder, kinds):
    names = _names(cfg)
    root = Path(cfg.text_encoder.cache_dir) / cfg.dataset / cfg.text_encoder.name
    tag = llm_tag(cfg)
    paths = {}
    for kind in kinds:
        if kind not in ('label', 'classwise', 'pairwise', 'pairwise_trunc'):
            raise ValueError(f'Unknown embedding kind: {kind}')
        frac = cfg.text_encoder.truncation_frac if kind == 'pairwise_trunc' else 1.0
        if kind == 'label':
            path = root / 'label.pt'
            texts = [f'a photo of a {name}' for name in names]
            digest = None
        else:
            source_kind = 'pairwise' if kind == 'pairwise_trunc' else kind
            source = Path(cfg.descriptions.cache_dir) / cfg.dataset / tag / f'{source_kind}.jsonl'
            check_manifest(cfg, source.parent)
            descriptions = load_descriptions(cfg.descriptions.cache_dir, cfg.dataset, tag, source_kind)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            filename = f'pairwise_trunc{frac:g}.pt' if kind == 'pairwise_trunc' else f'{kind}.pt'
            path = root / tag / filename
            if source_kind == 'pairwise':
                keys = sorted(required_pair_keys(data.seen_classes, data.unseen_classes, cfg.descriptions.symmetric))
            else:
                keys = list(range(len(names)))
            missing = set(keys) - descriptions.keys()
            if missing:
                raise ValueError(f'{source_kind} descriptions missing {len(missing)} required keys')
            texts = [truncate_text(descriptions[key], frac) if kind == 'pairwise_trunc' else descriptions[key] for key in keys]
        unique = list(dict.fromkeys(texts))
        vectors = encoder.encode(unique)
        lookup = {text: i for i, text in enumerate(unique)}
        if kind in ('pairwise', 'pairwise_trunc'):
            emb = torch.zeros((len(names), len(names), encoder.dim), dtype=torch.float32)
            for key, text in zip(keys, texts):
                a, b = key
                emb[a, b] = vectors[lookup[text]]
                if cfg.descriptions.symmetric:
                    emb[b, a] = vectors[lookup[text]]
        else:
            emb = vectors[[lookup[text] for text in texts]]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'classes': names, 'emb': emb.float().cpu()}, path)
        manifest = {'encoder': encoder.identity, 'symmetric': cfg.descriptions.symmetric,
                    'truncation_frac': frac, 'source_sha256': digest}
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        paths[kind] = path
    return paths
