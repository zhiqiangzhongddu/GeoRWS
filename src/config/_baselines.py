from yacs.config import CfgNode as CN


def set_baselines_cfg(cfg):
    cfg.baselines = CN()  # Configuration block
    cfg.baselines.subspace_proj = False  # Subspace projection
    cfg.baselines.daegnn = False  # Denoising graph network
    cfg.baselines.single_autoencoder_baseline = False  # Single autoencoder
    cfg.baselines.single_modal_ablation = False  # Single modality
    cfg.baselines.class_reduction_ablation = 0  # Class reduction
    cfg.baselines.vgse_baseline = ''  # VGSE variant
    cfg.baselines.vgse_nbs = 5  # Neighbours
    cfg.baselines.vgse_eta = 5  # Similarity scale
    cfg.baselines.vgse_alpha = 0.0  # SMO regularization
    cfg.baselines.conse_fixed_eval = False  # Correct ConSE unseen labels
