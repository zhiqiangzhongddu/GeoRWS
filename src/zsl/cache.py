from pathlib import Path


def base_classifier_path(cfg, seed, tag=''):
    b = cfg.base_classifier
    return Path(cfg.output_root) / 'base_classifiers' / (
        f'{cfg.dataset}_{cfg.zsl.image_embedding}{tag}_seed{seed}'
        f'_clr{b.lr}_b1{b.beta1}_bs{b.batch_size}_nep{b.num_epoch}.pt'
    )
