from yacs.config import CfgNode as CN


def set_base_classifier_cfg(cfg):
    cfg.base_classifier = CN()  # Configuration block
    cfg.base_classifier.lr = 0.0001  # Learning rate
    cfg.base_classifier.beta1 = 0.5  # Adam beta1
    cfg.base_classifier.num_epoch = 100  # Training epochs
    cfg.base_classifier.batch_size = 16  # Batch size
