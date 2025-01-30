import os
import random
import numpy as np
import argparse
import logging
from datetime import datetime
import time
import pytz
import torch
from torch import Tensor
import torch.nn as nn
from pathlib import PurePath


# * ============================= File-Path-Parameter Related (Start) =============================

project_root_path = PurePath(__file__).parent.parent.parent


# def is_notebook() -> bool:
#     from IPython import get_ipython
#     try:
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True  # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#
#     except NameError:
#         return False  # Probably standard Python interpreter


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_output_folder(args):
    folder = "../output/{}/{}_{}".format(
        args.dataset, args.model, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(folder)
    return folder


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_path(dir_or_file, is_dir=False):

    path = os.path.dirname(dir_or_file) + '/' if not is_dir else str(dir_or_file) + '/'

    if not os.path.exists(path):
        mkdir_p(path)

    return dir_or_file


def get_root_logger(folder):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if folder:
        handler = logging.FileHandler(os.path.join(folder, "log.txt"))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger

# * ============================= File-Path-Parameter Related (End) =============================


# * ============================= Time Related (Start) =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Europe/Paris', t_format='%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print('Start running {} at {}'.format(
            func.__name__, get_cur_time()
        ))
        ret = func(*args, **kw)
        print(
            'Finished running {} at {}, running time = {}.'.format(
                func.__name__, get_cur_time(), time2str(time.time() - start_time)
            ))
        return ret

    return wrapper

# * ============================= Time Related (End) =============================


# * ============================= Project Related (Start) =============================

class cos_sim_loss(nn.MSELoss):
    __constants__ = ['reduction']

    def __init__(self, dim = 1, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(cos_sim_loss, self).__init__(size_average, reduce, reduction)
        assert reduction in ['none', None, 'mean', 'sum']
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=dim, eps=1e-8)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = 1 - self.cos(input, target)
        if self.reduction == 'none' or self.reduction == None:
            return loss.unsqueeze(-1)
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


# Maps test classes to 151-200 instead of 1-50 (for latter, use map_label)
def map_label_extend(label, new_classes, base_classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(new_classes.size(0)):
        mapped_label[label == new_classes[i]] = i + len(base_classes)
    return mapped_label

# * ============================= Project Related (End) =============================
