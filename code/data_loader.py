import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

import code.data_loader_utils as data_loader_utils
import code.utility.utils as utils


def read_wiki2vec(cfg):
    """
    :param cfg:
    :return:
    """
    transfer_path = cfg.input_root + "/embeddings/wiki2vec/" + cfg.dataset + "_wiki_sum_list.npy"
    transfer_attributes = torch.from_numpy(np.load(transfer_path, allow_pickle=True)).float()
    transfer_attributes /= torch.norm(transfer_attributes, dim=1)[:, None]

    if cfg.zst_from == 'imagenet':
        source_path = cfg.input_root + "/embeddings/wiki2vec/imgnet_wiki_list.npy"
    else:
        source_path = cfg.input_root + "/embeddings/wiki2vec/" + cfg.zst_from + "_wiki_sum_list.npy"

    source_attributes = torch.from_numpy(np.load(source_path, allow_pickle=True)).float()
    source_attributes /= torch.norm(source_attributes, dim=1)[:, None]

    return transfer_attributes, source_attributes


def read_cn(cfg):
    """
    :param cfg:
    :return:
    """
    transfer_path = cfg.input_root + "/embeddings/conceptnet/" + cfg.dataset + "_cn_sum_list.npy"
    transfer_attributes = torch.from_numpy(np.load(transfer_path, allow_pickle=True)).float()
    transfer_attributes /= torch.norm(transfer_attributes, dim=1)[:, None]

    if cfg.zst_from == 'imagenet':
        source_path = cfg.input_root + "/embeddings/conceptnet/imgnet_cn_list.npy"
    else:
        source_path = cfg.input_root + "/embeddings/conceptnet/" + cfg.zst_from + "_cn_sum_list.npy"

    source_attributes = torch.from_numpy(np.load(source_path, allow_pickle=True)).float()
    source_attributes /= torch.norm(source_attributes, dim=1)[:, None]

    return transfer_attributes, source_attributes


def read_clip(cfg, mat_content):
    """
    :param cfg:
    :param mat_content:
    :return:
    """
    vocab = data_loader_utils.prepare_vocab(cfg, mat_content)
    transfer_attributes = data_loader_utils.get_clip_embeddings(vocab)
    if cfg.zst_from == 'imagenet':
        with open(cfg.data_root + "/ImageNet1K_classnames.txt") as f:
            in1k_classnames = f.read().splitlines()
        source_classnames = data_loader_utils.prep_imagenet_vocab(in1k_classnames)
    else:
        source_mat_content = sio.loadmat(
            cfg.data_root + "/" + cfg.zst_from + "/" + cfg.image_embedding + ".mat")
        source_label = source_mat_content['labels'].astype(int).squeeze() - 1
        source_path = cfg.data_root + "/" + cfg.zst_from + "/" + 'att' + "_splits.mat"
        source_mat_content = sio.loadmat(source_path)

        source_seen_loc = source_mat_content['test_seen_loc'].squeeze() - 1
        source_seen_classes = np.unique(source_label[source_seen_loc])

        source_classnames = data_loader_utils.prepare_vocab(cfg, source_mat_content, zst_mode=True)
        source_classnames = np.array(source_classnames)[source_seen_classes].tolist()

    source_attributes = data_loader_utils.get_clip_embeddings(source_classnames)

    return transfer_attributes, source_attributes


class MyDataLoader(object):
    """
    Data Loader
    """
    def __init__(self, cfg):

        # Class descriptor embeddings
        self.attribute = None

        # Number of training instances
        self.num_train = None

        # Labels of training instances
        self.train_label = None
        # Re-mapped labels of training instances
        self.train_mapped_label = None

        # Image embeddings
        self.train_feature = None
        self.test_unseen_feature = None
        self.test_unseen_label = None
        self.test_seen_feature = None
        self.test_seen_label = None

        # Set of classes
        self.unseen_classes = None
        self.seen_classes = None
        # Number of total classes
        self.num_class = 0

        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.read_mat_dataset(cfg)

    def read_mat_dataset(self, cfg):
        """
        :param cfg:
        """
        mat_content = sio.loadmat(cfg.data_root + "/" + cfg.dataset + "/" + cfg.cv.image_embedding + ".mat")

        feature = mat_content['features'].T
        # Embedding from Massi's feature extractor code (shape transposed compared to Yongqin features)
        if cfg.cv.image_embedding[:10] in ['pretrained']:
            feature = feature.T
        label = mat_content['labels'].astype(int).squeeze() - 1

        # Change class embedding (attributes or label embeddings)
        if cfg.cv.class_embedding in ['wiki2vec', 'cn', 'clip']:
            mat_path = cfg.data_root + "/" + cfg.dataset + "/" + 'att' + "_splits.mat"
        else:
            mat_path = cfg.data_root + "/" + cfg.dataset + "/" + cfg.cv.class_embedding + "_splits.mat"
        mat_content = sio.loadmat(mat_path)

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = mat_content['trainval_loc'].squeeze() - 1
        test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
        test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1

        if cfg.zst:
            if cfg.cv.class_embedding == 'wiki2vec':
                source_attributes, transfer_attributes = read_wiki2vec(cfg=cfg)
            elif cfg.cv.class_embedding == 'cn':
                source_attributes, transfer_attributes = read_cn(cfg=cfg)
            elif cfg.cv.class_embedding == 'clip':
                source_attributes, transfer_attributes = read_clip(cfg=cfg, mat_content=mat_content)
            else:
                raise NotImplementedError
            self.attribute = torch.cat((source_attributes, transfer_attributes))
        else:
            source_attributes, transfer_attributes = None, None
            if cfg.cv.class_embedding == 'wiki2vec':
                embedding_path = cfg.input_root + "/embeddings/wiki2vec/" + cfg.dataset + "_wiki_sum_list.npy"
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
            elif cfg.cv.class_embedding == 'cn':
                embedding_path = cfg.input_root + "/embeddings/conceptnet/" + cfg.dataset + "_cn_sum_list.npy"
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
            elif cfg.cv.class_embedding == 'clip':
                vocab = data_loader_utils.prepare_vocab(cfg, mat_content)
                self.attribute = data_loader_utils.get_clip_embeddings(vocab)
            elif cfg.cv.class_embedding == 'att':
                self.attribute = torch.from_numpy(mat_content['att'].T).float()
            else:
                raise NotImplementedError

        print("Loaded attributes / word embeddings with shape", self.attribute.shape)

        self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seen_classes = torch.unique(self.train_label)
        self.unseen_classes = torch.unique(self.test_unseen_label)

        if cfg.zst:
            self.unseen_classes += len(source_attributes)
            self.seen_classes = torch.arange(len(source_attributes))
        else:
            pass

        self.num_train = self.train_feature.size()[0]
        self.num_class = len(self.seen_classes) + len(self.unseen_classes)
        self.train_mapped_label = utils.map_label(self.train_label, self.seen_classes)


class GenericDataset(Dataset):
    def __init__(self, cfg, _input, _target, cuda, transform = None):
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
