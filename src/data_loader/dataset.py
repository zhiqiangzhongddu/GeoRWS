from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from src.data_loader.class_names import class_names, get_clip_embeddings
from src.utils.naming import llm_tag
from src.zsl.utils import map_label


class ZSLDataset:
    def __init__(self, cfg):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.read_mat_dataset(cfg)

    def read_mat_dataset(self, cfg):
        root = Path(cfg.data_root)
        mat_content = sio.loadmat(root / cfg.dataset / (cfg.zsl.image_embedding + '.mat'))
        feature = mat_content['features'].T
        if cfg.zsl.image_embedding[:10] in ['pretrained']:
            feature = feature.T
        label = mat_content['labels'].astype(int).squeeze() - 1
        mat_content = sio.loadmat(root / cfg.dataset / 'att_splits.mat')
        self.class_names = class_names(cfg.dataset, mat_content)
        trainval_loc = mat_content['trainval_loc'].reshape(-1) - 1
        train_loc = mat_content['train_loc'].reshape(-1) - 1
        val_loc = mat_content['val_loc'].reshape(-1) - 1

        embedding = cfg.zsl.class_embedding
        if embedding in ('wiki2vec', 'cn'):
            folder, suffix = ('wiki2vec', 'wiki') if embedding == 'wiki2vec' else ('conceptnet', 'cn')
            path = root / 'class_embeddings' / folder / f'{cfg.dataset}_{suffix}_sum_list.npy'
            self.attribute = torch.from_numpy(np.load(path, allow_pickle=True)).float()
            self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
        elif embedding == 'clip':
            self.attribute = get_clip_embeddings(self.class_names)
        elif embedding == 'att':
            self.attribute = torch.from_numpy(mat_content['att'].T).float()
        elif embedding == 'llm':
            path = Path(cfg.text_encoder.cache_dir) / cfg.dataset / cfg.text_encoder.name / llm_tag(cfg) / 'classwise.pt'
            if not path.is_file():
                raise FileNotFoundError(f'Classwise embeddings not found: {path}')
            self.attribute = torch.load(path, map_location='cpu', weights_only=True)['emb'].float()
        else:
            raise NotImplementedError(embedding)
        print('Loaded attributes / word embeddings with shape', self.attribute.shape)
        self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.seen_classes = torch.unique(self.train_label)
        if cfg.zsl.mode == 'validate':
            # Validation uses held-out trainval images and never selects test examples.
            self.test_unseen_feature = self.test_unseen_label = None
            self.test_seen_feature = self.test_seen_label = None
            all_classes = torch.arange(self.attribute.size(0))
            self.unseen_classes = all_classes[~torch.isin(all_classes, self.seen_classes)]
        else:
            test_seen_loc = mat_content['test_seen_loc'].reshape(-1) - 1
            test_unseen_loc = mat_content['test_unseen_loc'].reshape(-1) - 1
            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            self.unseen_classes = torch.unique(self.test_unseen_label)
        self.train_classes = torch.unique(torch.from_numpy(label[train_loc]).long())
        self.val_classes = torch.unique(torch.from_numpy(label[val_loc]).long())
        assert not torch.isin(self.train_classes, self.val_classes).any()
        assert torch.equal(torch.sort(torch.cat((self.train_classes, self.val_classes))).values, self.seen_classes)
        self.num_train = self.train_feature.size(0)
        self.num_class = len(self.seen_classes) + len(self.unseen_classes)
        self.train_mapped_label = map_label(self.train_label, self.seen_classes)


class GenericDataset(Dataset):
    def __init__(self, cfg, _input, _target, cuda, transform=None):
        assert len(_input) == len(_target)
        self.cfg = cfg
        self.input = _input
        self.target = _target
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        in_var = self.input[idx]
        target = self.target[idx]
        if self.cuda:
            in_var = in_var.cuda()
            target = target.cuda()
        if self.transform:
            in_var = self.transform(in_var)
        return in_var, target
