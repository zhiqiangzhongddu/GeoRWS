from yacs.config import CfgNode as CN


def set_train_cfg(cfg):
    cfg.train = CN()  # Configuration block
    cfg.train.lr = 1e-05  # Learning rate
    cfg.train.beta1 = 0.9  # Adam beta1
    cfg.train.num_epoch = 1000  # Training epochs
    cfg.train.batch_size = 16  # Batch size
    cfg.train.embed_dim = 2048  # Embedding dimension
    cfg.train.num_layers = 2  # Layer count
    cfg.train.early_stopping_slope = True  # Use loss slope stopping
    cfg.train.cos_sim_loss = False  # Use MSE by default; ICIS opts into cosine loss
    cfg.train.include_unseen = False  # ICIS opts into including unseen attributes
