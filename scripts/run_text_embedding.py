"""Build cached text embeddings for available descriptions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import cfg, update_cfg
from src.data_loader.dataset import ZSLDataset
from src.descriptions.embed import build_embeddings
from src.descriptions.encoders import make_text_encoder
from src.utils.naming import llm_tag


def run(argv=None):
    config = update_cfg(cfg, argv)
    data = ZSLDataset(config)
    encoder = make_text_encoder(config)
    directory = Path(config.descriptions.cache_dir) / config.dataset / llm_tag(config)
    kinds = ['label']
    for kind in ('classwise', 'pairwise'):
        if (directory / f'{kind}.jsonl').exists():
            kinds.append(kind)
            if kind == 'pairwise' and config.text_encoder.truncation_frac < 1:
                kinds.append('pairwise_trunc')
        else:
            print(f'Skipping {kind}: {directory / (kind + ".jsonl")} not found')
    for path in build_embeddings(config, data, encoder, kinds).values():
        print(f'Wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:]))
