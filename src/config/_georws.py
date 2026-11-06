from yacs.config import CfgNode as CN


def set_georws_cfg(cfg):
    cfg.georws = CN()  # Configuration block
    cfg.georws.latent_dim = 1000  # Latent dimension
    cfg.georws.hidden_dim = 1000  # Hidden dimension
    cfg.georws.lam = 1.0  # Reconstruction weight
    cfg.georws.beta = 0.1  # Geometry weight
    cfg.georws.gamma = 0.1  # Consistency weight
    cfg.georws.tau_w = 0.1  # Teacher temperature
    cfg.georws.aug_dropout = 0.1  # Augmentation dropout
    cfg.georws.use_geo = True  # Enable geometry loss
    cfg.georws.use_cons = True  # Enable consistency loss
    cfg.georws.adaptive_encoder = True  # Enable encoder
    cfg.georws.semantic_source = 'pairwise'  # pairwise | classwise | label
    cfg.georws.recon_reduction = 'sum'  # sum | mean
    cfg.georws.unseen_scale = 'seen_norm'  # seen_norm | none | icis
    cfg.georws.early_stopping = False  # Enable early stopping
    cfg.georws.early_stopping_threshold = 0.0001  # Loss slope threshold
    cfg.georws.log_every = 50  # Logging interval
    cfg.georws.infer_batch_size = 64  # Inference batch size
    cfg.georws.perturb = CN()  # Configuration block
    cfg.georws.perturb.kind = 'none'  # none | embedding_dropout | gaussian_noise | text_truncation | pair_swap
    cfg.georws.perturb.rate = 0.2  # Perturbation rate
    cfg.georws.perturb.sigma = 0.2  # Noise ratio
    cfg.georws.perturb.truncation_frac = 0.5  # Text fraction
    cfg.georws.perturb.seed = 0  # Perturbation seed
    cfg.georws.val = CN()  # Configuration block
    cfg.georws.val.holdout_frac = 0.2  # Holdout fraction
    cfg.georws.val.seed = 0  # Validation seed
