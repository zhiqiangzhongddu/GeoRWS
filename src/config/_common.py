from yacs.config import CfgNode as CN


def set_general_cfg(cfg):
    cfg.device = 0  # CUDA device; negative selects CPU
    cfg.seeds = [42, 0, 100, 123, 2024]  # Ordered run seeds
    cfg.num_runs = 1  # Number of seeds
    cfg.dataset = 'CUB'  # Dataset name
    cfg.method = 'georws'  # Method name
    cfg.data_root = 'data'  # Data directory
    cfg.output_root = 'outputs'  # Output directory
