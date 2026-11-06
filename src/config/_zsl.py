from yacs.config import CfgNode as CN


def set_zsl_cfg(cfg):
    cfg.zsl = CN()  # Configuration block
    cfg.zsl.image_embedding = 'res101_finetuned'  # Image features
    cfg.zsl.class_embedding = 'att'  # att | wiki2vec | cn | clip | llm
    cfg.zsl.strict_eval = True  # Evaluate after training
    cfg.zsl.mode = 'test'  # test | validate
    cfg.zsl.eval_every = 0  # 0 = never; >0 diagnostic only
    cfg.zsl.save_pred_matrix = False  # Save predictions
    cfg.zsl.calc_entropy = False  # Calculate entropy
    cfg.zsl.norm_scale_heuristic = False  # ICIS norm scaling
    cfg.zsl.seen_logit_bias = 0.0  # Calibrated stacking for GeoRWS
