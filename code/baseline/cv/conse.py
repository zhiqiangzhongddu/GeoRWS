import torch
import torch.nn as nn
from torch.autograd import Variable
import code.utility.utils as utils
from code.utility.regressor import REGRESSOR


class ConSE(REGRESSOR):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.cfg = cfg
        if self.cuda:
            self.reg_model.cuda()

        if self.cfg.zst:
            data = self.test_unseen_feature
            target = self.test_unseen_label

            self.acc_target = self.conse_val(
                self.reg_model, data,
                utils.map_label(target, self.unseen_classes - len(self.seen_classes)),
                utils.map_label(self.unseen_classes - len(self.seen_classes),
                                self.unseen_classes - len(self.seen_classes)),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=self.attribute[self.unseen_classes]
            )

            self.acc_zst_unseen = self.conse_val(
                self.reg_model, data,
                utils.map_label(target,
                                self.unseen_classes - len(self.seen_classes)) + len(
                    self.seen_classes),
                utils.map_label_extend(self.unseen_classes, self.unseen_classes,
                                       self.seen_classes),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=torch.cat((self.attribute[self.seen_classes],
                                           self.attribute[self.unseen_classes]))
            )

        else:
            # GZSL
            self.acc_gzsl = self.conse_val(
                self.reg_model, torch.cat((self.test_seen_feature, self.test_unseen_feature), 0),
                torch.cat((utils.map_label(self.test_seen_label, self.seen_classes),
                           utils.map_label_extend(self.test_unseen_label, self.unseen_classes,
                                                  self.seen_classes)), 0),
                torch.cat((utils.map_label(self.seen_classes, self.seen_classes),
                           utils.map_label_extend(self.unseen_classes, self.unseen_classes,
                                                  self.seen_classes)), 0),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=torch.cat((self.attribute[self.seen_classes],
                                           self.attribute[self.unseen_classes]))
            )
            self.acc_seen = self.conse_val(
                self.reg_model, self.test_seen_feature,
                utils.map_label(self.test_seen_label, self.seen_classes),
                utils.map_label(self.seen_classes, self.seen_classes),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=torch.cat((self.attribute[self.seen_classes],
                                           self.attribute[self.unseen_classes]))
            )
            self.acc_unseen = self.conse_val(
                self.reg_model, self.test_unseen_feature,
                utils.map_label(self.test_unseen_label, self.unseen_classes),
                utils.map_label(self.unseen_classes, self.unseen_classes),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=torch.cat((self.attribute[self.seen_classes],
                                           self.attribute[self.unseen_classes]))
            )
            self.H = 2 * self.acc_seen * self.acc_unseen / (self.acc_seen + self.acc_unseen)
            # ZSL 
            self.acc_unseen_zsl = self.conse_val(
                self.reg_model, self.test_unseen_feature,
                utils.map_label(self.test_unseen_label, self.unseen_classes),
                utils.map_label(self.unseen_classes, self.unseen_classes),
                train_attributes=self.attribute[self.seen_classes],
                test_attributes=self.attribute[self.unseen_classes]
            )

    def conse_val(
            self, model, test_X, test_label, target_classes, train_attributes, test_attributes
    ):
        """ Predict semantic embedding for input, then compare to class embeddings (attributes) """
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        soft = torch.nn.Softmax(dim=1)
        if self.cuda:
            train_attributes = train_attributes.cuda()
            test_attributes = test_attributes.cuda()
        start = 0
        num_test = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, num_test, self.batch_size):
            end = min(num_test, start + self.batch_size)
            if self.cuda:
                logits = model(Variable(test_X[start:end].cuda()))
            else:
                logits = model(Variable(test_X[start:end]))

            if self.cfg.cv.class_reduction_ablation:
                probs = soft(logits[:, self.perm])
                pred_embeds = torch.sum(train_attributes[self.perm] * probs.unsqueeze(-1), dim=1)
            else:
                probs = soft(logits)
                pred_embeds = torch.sum(train_attributes * probs.unsqueeze(-1), dim=1)

            output = []
            for pred_embed in pred_embeds:
                sims = cos(pred_embed[None, :], test_attributes)
                _, idx = torch.max(sims, dim=0)
                output.append(idx)

            output = torch.stack(output)
            predicted_label[start:end] = output
            start = end

        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label,
            target_classes
        )
        if self.cfg.save_pred_matrix:
            torch.save(
                acc_per_class,
                self.cfg.out_root + '/predictions/' + self.cfg.dataset + self.cfg.image_embedding
                + '_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt'
            )
            torch.save(
                prediction_matrix,
                self.cfg.out_root + '/predictions/' + self.cfg.dataset + self.cfg.image_embedding
                + '_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt'
            )

        return acc
