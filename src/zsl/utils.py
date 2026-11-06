import torch
from torch import Tensor
import torch.nn as nn
from src.utils.random import set_seed


def use_cuda(cfg):
    return cfg.device >= 0 and torch.cuda.is_available()


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


def harmonic_mean(acc_seen, acc_unseen):
    """GZSL H = 2su/(s+u); defined as 0 when both accuracies are 0 (the legacy formula gave NaN)."""
    denominator = acc_seen + acc_unseen
    if float(denominator) == 0.0:
        return denominator * 0
    return 2 * acc_seen * acc_unseen / denominator
