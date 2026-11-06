from yacs.config import CfgNode as CN


def set_save_results_cfg(cfg):
    cfg.save_results = CN()  # Configuration block
    cfg.save_results.enabled = True  # Save result rows
    cfg.save_results.output_dir = 'outputs/results'  # Result directory
