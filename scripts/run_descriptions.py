"""Generate descriptions.

python scripts/run_descriptions.py dataset CUB llm.name gpt-4o descriptions.kind pairwise
Dry run: python scripts/run_descriptions.py llm.provider fake descriptions.max_items 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import cfg, update_cfg
from src.data_loader.dataset import ZSLDataset
from src.descriptions.generate import generate_descriptions
from src.descriptions.llm import make_chat_client


def run(argv=None):
    config = update_cfg(cfg, argv)
    data = ZSLDataset(config)
    client = make_chat_client(config.llm)
    generate_descriptions(config, data, client, config.descriptions.kind)
    return 0


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:]))
