from types import SimpleNamespace

import torch

from src.zsl.run import prepare_base_model
from src.zsl.utils import set_seed


def validate_georws(cfg, data, seed):
    from src.georws.run import fit_and_evaluate

    train, val = data.train_classes, data.val_classes
    generator = torch.Generator().manual_seed(cfg.georws.val.seed)
    fit_indices, held_indices = [], []
    for cls in train:
        indices = (data.train_label == cls).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n = max(1, round(cfg.georws.val.holdout_frac * len(indices)))
        held_indices.append(indices[:n])
        fit_indices.append(indices[n:])
    fit_indices = torch.cat(fit_indices)
    held_indices = torch.cat(held_indices)
    val_indices = torch.isin(data.train_label, val).nonzero(as_tuple=True)[0]
    view = SimpleNamespace(
        seen_classes=train, unseen_classes=val, train_classes=train, val_classes=val,
        train_feature=data.train_feature[fit_indices], train_label=data.train_label[fit_indices],
        test_seen_feature=data.train_feature[held_indices], test_seen_label=data.train_label[held_indices],
        test_unseen_feature=data.train_feature[val_indices], test_unseen_label=data.train_label[val_indices],
        attribute=data.attribute, class_names=data.class_names, num_class=len(train) + len(val))
    # Cache the head separately for each validation split.
    tag = f'_val_s{cfg.georws.val.seed}_h{cfg.georws.val.holdout_frac:g}'
    path = prepare_base_model(cfg, view, seed, tag=tag)
    state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    weights = torch.cat((state['fc.weight'], state['fc.bias'][:, None]), dim=1)
    set_seed(seed)
    result = fit_and_evaluate(cfg, view, seed, target_weights=weights)
    print(f'validation: {result}')
    return result
