import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import code.utility.utils as utils
import copy
from code.utility import model_bases
from code.utility.regressor import REGRESSOR


class BASECLASSIFIER(REGRESSOR):
    def __init__(
            self, _train_x, _train_y, data_loader, _num_class, seed,
            _cuda = True, train_base = True, _lr = 0.001,
            _beta1 = 0.5, _num_epoch = 20, _batch_size = 100, _embed_dim = 1000, _num_layers = 3, cfg = None
    ):
        super().__init__(
            _train_x=_train_x, _train_y=_train_y, data_loader=data_loader, _num_class=_num_class, _cuda=_cuda,
            seed=seed, train_base=train_base, _lr=_lr, _beta1=_beta1, _num_epoch=_num_epoch,
            _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, cfg=cfg
        )
        self.cfg = cfg
        self.seed = seed

        self.num_epoch = _num_epoch
        self.reg_model = model_bases.LINEAR(self.input_dim, len(self.seen_classes))
        self.reg_model.apply(utils.weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_classifier = optim.Adam(
            self.reg_model.parameters(), lr=self.cfg.classifier_lr,
            betas=(self.cfg.classifier_beta1, 0.999)
        )

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        if self.cuda:
            self.reg_model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.num_train = self.train_x.size()[0]

    def fit(self):
        best_seen = 0
        best_model = copy.deepcopy(self.reg_model)

        # if self.train_base:
        for epoch in range(self.num_epoch):
            for i in range(0, self.num_train, self.batch_size):
                self.reg_model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.reg_model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer_classifier.step()

            acc_train = self.val_model(
                self.reg_model, self.train_x, self.train_y,
                utils.map_label(self.seen_classes, self.seen_classes)
            )
            acc_val_seen = self.val_model(
                self.reg_model, self.test_seen_feature,
                utils.map_label(self.test_seen_label, self.seen_classes),
                utils.map_label(self.seen_classes, self.seen_classes)
            )

            if acc_val_seen > best_seen:
                print(
                    f'New best validation seen class accuracy={acc_val_seen * 100:.4f}% '
                    f'(train seen class accuracy={acc_train * 100:.4f}%)'
                )
                best_seen = acc_val_seen
                best_model = copy.deepcopy(self.reg_model)
            else:
                pass

        return best_model

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.num_train)
            self.train_x = self.train_x[perm]
            self.train_y = self.train_y[perm]
        # the last batch
        if start + batch_size > self.num_train:
            self.epochs_completed += 1
            rest_num_examples = self.num_train - start
            if rest_num_examples > 0:
                x_rest_part = self.train_x[start:self.num_train]
                y_rest_part = self.train_y[start:self.num_train]
            else:
                x_rest_part, y_rest_part = None, None
            # shuffle the data
            perm = torch.randperm(self.num_train)
            self.train_x = self.train_x[perm]
            self.train_y = self.train_y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_x[start:end]
            Y_new_part = self.train_y[start:end]
            if rest_num_examples > 0:
                return torch.cat((x_rest_part, X_new_part), 0), torch.cat((y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_x[start:end], self.train_y[start:end]
