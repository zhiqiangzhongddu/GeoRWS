from code.utility.regressor import REGRESSOR


class Baseline(REGRESSOR):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)

    def evaluate_weights(self, pred_weights):
        self.unseen_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim]
        self.unseen_model.fc.bias.data[:] = pred_weights[:, self.input_dim]

        self.ext_model.fc.weight.data[len(self.seen_classes):, :] = pred_weights[:, :self.input_dim]
        self.ext_model.fc.bias.data[len(self.seen_classes):] = pred_weights[:, self.input_dim]

        if self.cfg.zst:
            self.acc_target, self.acc_zst_unseen = self.val_zst()
        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = self.val_gzsl()
