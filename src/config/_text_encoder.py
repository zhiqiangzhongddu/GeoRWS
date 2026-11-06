from yacs.config import CfgNode as CN


def set_text_encoder_cfg(cfg):
    cfg.text_encoder = CN()  # Configuration block
    cfg.text_encoder.name = 'clip'  # clip | bert | t5 | fake
    cfg.text_encoder.batch_size = 64  # Batch size
    cfg.text_encoder.cache_dir = 'data/embeddings'  # Embedding cache
    cfg.text_encoder.truncation_frac = 1.0  # Fraction of text retained
