from yacs.config import CfgNode as CN


def set_descriptions_cfg(cfg):
    cfg.descriptions = CN()  # Configuration block
    cfg.descriptions.kind = 'pairwise'  # pairwise | classwise
    cfg.descriptions.symmetric = False  # Use unordered pairs
    cfg.descriptions.cache_dir = 'data/descriptions'  # Description cache
    cfg.descriptions.max_items = 0  # 0 = no limit
