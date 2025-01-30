import os
import random
import torch

from code.data_loader import MyDataLoader
import code.baseline.cv.conse as conse
import code.baseline.cv.vgse as vgse
import code.baseline.cv.costa as costa
import code.utility.utils as utils
from code.utility import joint_latent
from code.utility.classifier import BASECLASSIFIER


def prepare_base_model(cfg, data, seed):
    # load or train base classification model
    if cfg.zst:
        pass
    else:
        if not os.path.exists(cfg.out_root + '/models/base-classifiers/'):
            os.makedirs(cfg.out_root + '/models/base-classifiers/')
        model_path = (cfg.out_root + '/models/base-classifiers/' + cfg.dataset + cfg.cv.image_embedding
                      + f'_seed{seed}_clr{cfg.classifier_lr}_nep{cfg.classifier_num_epoch}')
        if os.path.isfile(model_path):
            print(
                f"Existing base classifier for dataset {cfg.dataset} on seed {seed} "
                f"with given classifier training settings detected. Loading model and skipping training."
            )
        else:
            base_model = BASECLASSIFIER(
                _train_x=data.train_feature,
                _train_y=utils.map_label(data.train_label, data.seen_classes),
                data_loader=data, _num_class=data.num_class, seed=seed,
                _lr=cfg.lr, _beta1=cfg.beta1, _num_epoch=cfg.classifier_num_epoch,
                _batch_size=cfg.batch_size, _embed_dim=cfg.embed_dim, _num_layers=cfg.num_layers,
                cfg=cfg
            ).fit()
            torch.save(base_model, model_path)
            print(f"Saved base classifier for dataset {cfg.dataset} trained on seed {seed}.")


def run_cv_experiments(cfg):

    seed_list = [cfg.seed] if cfg.num_runs == 1 \
        else [random.randint(1, 10000) for _ in range(cfg.num_runs)]

    accs_unseen_only, accs_gzsl, accs_unseen, accs_seen, hs = [], [], [], [], []
    accs_unseen_only_std, accs_gzsl_std, accs_unseen_std, accs_seen_std, hs_std = [], [], [], [], []

    acc_gzsl_seeds_avg, acc_seen_seeds_avg, acc_unseen_seeds_avg, H_seeds_avg, unseen_zsl_seeds_avg =\
        [], [], [], [], []

    for seed in seed_list:
        utils.set_seed(seed=seed)

        # load data
        data = MyDataLoader(cfg)

        # prepare base classification model
        prepare_base_model(cfg=cfg, data=data, seed=seed)

        # Run different methods
        if cfg.method in ["wavg", "smo"]:
            assert cfg.cv.vgse_baseline in ["wavg", "smo"]
            model = vgse.VGSE_CRM(
                _train_x=data.train_feature,
                _train_y=utils.map_label(data.train_label, data.seen_classes),
                data_loader=data,
                _num_class=data.num_class, seed=seed,
                _lr=cfg.lr, _beta1=cfg.beta1, _num_epoch=cfg.num_epoch, _batch_size=cfg.batch_size,
                _embed_dim=cfg.embed_dim, _num_layers=cfg.num_layers, cfg=cfg
            )

        elif cfg.method == "conse":
            assert cfg.cv.conse_benchmark is True
            bs = cfg.batch_size
            model = conse.ConSE(
                _train_x=data.train_feature,
                _train_y=utils.map_label(data.train_label, data.seen_classes),
                data_loader=data,
                _num_class=data.num_class, seed=seed,
                _lr=cfg.lr, _beta1=cfg.beta1, _num_epoch=cfg.num_epoch, _batch_size=bs, _embed_dim=cfg.embed_dim,
                _num_layers=cfg.num_layers, cfg=cfg
            )

        elif cfg.method == "costa":
            assert cfg.cv.costa_benchmark is True
            model = costa.COSTA(
                _train_x=data.train_feature,
                _train_y=utils.map_label(data.train_label, data.seen_classes), data_loader=data,
                _num_class=data.num_class, seed=seed,
                _lr=cfg.lr, _beta1=cfg.beta1, _num_epoch=cfg.num_epoch, _batch_size=cfg.batch_size,
                _embed_dim=cfg.embed_dim, _num_layers=cfg.num_layers, cfg=cfg
            )

        elif cfg.method in ["icis", "subreg", "wdae"]:
            model = joint_latent.Joint(
                _train_x=data.train_feature,
                _train_y=utils.map_label(data.train_label, data.seen_classes), data_loader=data,
                _num_class=data.num_class, seed=seed,
                _lr=cfg.lr, _beta1=cfg.beta1, _num_epoch=cfg.num_epoch, _batch_size=cfg.batch_size,
                _embed_dim=cfg.embed_dim, _num_layers=cfg.num_layers, cfg=cfg
            )
            model.fit()

        else:
            raise NotImplementedError

        if cfg.zst:
            acc_unseen_only, acc_unseen = model.acc_target, model.acc_zst_unseen
            print(f"I-ZSL accuracy from {cfg.zst_from} transfer: {acc_unseen_only * 100:.2f}%.")
            print(f"Unseen accuracy (not H) I-GZSL from {cfg.zst_from} transfer: {acc_unseen * 100:.2f}%.")
        else:
            acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_only \
                = model.acc_gzsl, model.acc_seen, model.acc_unseen, model.H, model.acc_unseen_zsl
            print(f"I-ZSL (unseen only) Acc = {acc_unseen_only * 100:.2f}%")
            print(
                f"I-GZSL (seen and unseen): "
                f"H = {H * 100:.2f}, "
                f"Seen={acc_seen * 100:.2f}%, "
                f"Unseen={acc_unseen * 100:.2f}%"
            )
        print("-------")

        unseen_zsl_seeds_avg.append(acc_unseen_only)
        acc_unseen_seeds_avg.append(acc_unseen)
        if cfg.zst:
            pass
        else:
            acc_gzsl_seeds_avg.append(acc_gzsl)
            acc_seen_seeds_avg.append(acc_seen)
            H_seeds_avg.append(H)

        accs_unseen_only.append(torch.std_mean(torch.stack(unseen_zsl_seeds_avg), dim=0, unbiased=False)[1])
        accs_unseen_only_std.append(torch.std_mean(torch.stack(unseen_zsl_seeds_avg), dim=0, unbiased=False)[0])
        accs_unseen.append(torch.std_mean(torch.stack(acc_unseen_seeds_avg), dim=0, unbiased=False)[1])
        accs_unseen_std.append(torch.std_mean(torch.stack(acc_unseen_seeds_avg), dim=0, unbiased=False)[0])
        if cfg.zst:
            pass
        else:
            accs_gzsl.append(torch.std_mean(torch.stack(acc_gzsl_seeds_avg), dim=0, unbiased=False)[1])
            accs_gzsl_std.append(torch.std_mean(torch.stack(acc_gzsl_seeds_avg), dim=0, unbiased=False)[0])
            accs_seen.append(torch.std_mean(torch.stack(acc_seen_seeds_avg), dim=0, unbiased=False)[1])
            accs_seen_std.append(torch.std_mean(torch.stack(acc_seen_seeds_avg), dim=0, unbiased=False)[0])
            hs.append(torch.std_mean(torch.stack(H_seeds_avg), dim=0, unbiased=False)[1])
            hs_std.append(torch.std_mean(torch.stack(H_seeds_avg), dim=0, unbiased=False)[0])

    # Collect all runs' results
    accs_unseen_only = torch.stack(accs_unseen_only)
    accs_unseen_only_std = torch.stack(accs_unseen_only_std)
    accs_unseen = torch.stack(accs_unseen)
    accs_unseen_std = torch.stack(accs_unseen_std)
    idx_best_unseen = torch.argmax(accs_unseen)
    if cfg.zst:
        pass
    else:
        accs_seen = torch.stack(accs_seen)
        accs_seen_std = torch.stack(accs_seen_std)
        hs = torch.stack(hs)
        hs_std = torch.stack(hs_std)

        idx_best_H = torch.argmax(hs)

    if cfg.num_runs > 1:
        if cfg.zst:
            print(
                f"Performance, mean over seeds: \n"
                f"I-ZSL (unseen only) "
                f"Acc = ({accs_unseen_only[idx_best_unseen] * 100:.2f} "
                f"+/- {accs_unseen_only_std[idx_best_unseen] * 100:.2f})% \n "
                f"I-GZSL Unseen "
                f"Acc = ({accs_unseen[idx_best_H] * 100:.2f} "
                f"+/- {accs_unseen_std[idx_best_H] * 100:.2f})% \n"
                f"Averaged over seeds {seed_list}. "
                f"For Seen accuracy (and thus H), evaluate using eval_imagenet.py"
            )
        else:
            print(
                f"Performance, mean over seeds: \n"
                f"I-ZSL (unseen only) "
                f"Acc = ({accs_unseen_only[idx_best_H] * 100:.2f} "
                f"+/- {accs_unseen_only_std[idx_best_H] * 100:.2f})% \n"
                f"I-GZSL (seen and unseen) "
                f"H = ({hs[idx_best_H] * 100:.2f} +/- {hs_std[idx_best_H] * 100:.2f}), "
                f"Unseen Acc = ({accs_unseen[idx_best_H] * 100:.2f} "
                f"+/- {accs_unseen_std[idx_best_H] * 100:.2f})%, "
                f"Seen Acc = ({accs_seen[idx_best_H] * 100:.2f} "
                f"+/- {accs_seen_std[idx_best_H] * 100:.2f})% \n"
                f"Averaged over seeds {seed_list}"
            )

        print("All experiments over the list of seeds completed.")
        print("-------------------------")
