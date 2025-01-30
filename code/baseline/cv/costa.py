import torch
from code.utility import model_bases
from code.utility.regressor import REGRESSOR


class COSTA(REGRESSOR):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.cfg = cfg

        self.unseen_model = model_bases.LINEAR(self.input_dim, len(self.unseen_classes))
        self.ext_model = model_bases.LINEAR(self.input_dim, self.num_class)
        if self.cuda:
            self.unseen_model.cuda()
            self.ext_model.cuda()

        self.ext_model.fc.weight.data[:len(self.seen_classes), :] = self.target_weights[:, :2048]
        self.ext_model.fc.bias.data[:len(self.seen_classes)] = self.target_weights[:, 2048]
        for n, unseen_att in enumerate(self.attribute[self.unseen_classes]):
            cooccs = unseen_att.unsqueeze(0) * self.attribute[self.seen_classes]
            norm_coocs = torch.sum(cooccs, dim=-1) / (cooccs.sum() + 10e-5)
            if self.cuda:
                norm_coocs = norm_coocs.cuda()
            pred_weights = torch.sum(norm_coocs[:, None] * self.target_weights, dim=0)

            self.unseen_model.fc.weight.data[n, :] = pred_weights[:-1]
            self.unseen_model.fc.bias.data[n] = pred_weights[-1]

            self.ext_model.fc.weight.data[len(self.seen_classes) + n, :] = pred_weights[:-1]
            self.ext_model.fc.bias.data[len(self.seen_classes) + n] = pred_weights[-1]

        # GZSL
        if self.cfg.zst:
            self.acc_target, self.acc_zst_unseen = self.val_zst()

        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = self.val_gzsl()
