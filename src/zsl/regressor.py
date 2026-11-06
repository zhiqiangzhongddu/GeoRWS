import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import src.zsl.utils as utils
from src.zsl import model_bases
from src.zsl.cache import base_classifier_path


class REGRESSOR:
    def __init__(
            self, _train_x, _train_y, data_loader, _num_class, seed, cfg,
            _cuda = True, train_base = False, _lr = 0.001,
            _beta1 = 0.5, _num_epoch = 20, _batch_size = 100, _embed_dim = 1000, _num_layers = 3, target_weights = None,
    ):
        self.ref_weights = None
        self.train_x = _train_x
        self.train_y = _train_y

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seen_classes = data_loader.seen_classes
        self.unseen_classes = data_loader.unseen_classes

        self.attribute = data_loader.attribute
        self.batch_size = _batch_size
        self.num_epoch = _num_epoch
        self.num_class = _num_class
        self.embed_dim = _embed_dim
        self.num_layers = _num_layers
        self.input_dim = _train_x.size(1)
        self.cuda = _cuda
        self.reg_model = model_bases.LINEAR(self.input_dim, len(self.seen_classes))
        self.reg_model.apply(utils.weights_init)
        self.criterion = nn.CrossEntropyLoss()

        # To be defined depending on different methods
        self.unseen_model = None
        self.ext_model = None

        self.cfg = cfg
        self.seed = seed

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer_classifier = optim.Adam(
            self.reg_model.parameters(),
            lr=self.cfg.base_classifier.lr,
            betas=(self.cfg.base_classifier.beta1, 0.999)
        )
        self.calc_entropy = False

        if self.cuda:
            self.reg_model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.num_train = self.train_x.size()[0]

        if not train_base:
            if target_weights is None:
                checkpoint = torch.load(base_classifier_path(cfg, seed), map_location='cpu', weights_only=True)
                with torch.random.fork_rng(devices=[]):
                    best_model = model_bases.LINEAR(checkpoint['input_dim'], checkpoint['num_classes'])
                best_model.load_state_dict(checkpoint['state_dict'])
                if self.cuda:
                    best_model.cuda()
                self.reg_model = best_model
                self.target_weights = torch.cat(
                    (best_model.fc.weight.data, torch.unsqueeze(best_model.fc.bias.data, 1)), 1
                )
            else:
                self.target_weights = target_weights.to('cuda' if self.cuda else 'cpu')
            self.ref_norm = torch.norm(self.target_weights, dim=-1).mean()

    def pred_weights_and_val(self, weight_model, daegnn = None):
        """ Predict weights and insert in extended GZSL model and/or ZSL model. Then evaluate performance. """
        attributes_to_regress = self.attribute[self.unseen_classes]
        for n, attribute_vector in enumerate(attributes_to_regress):
            attribute_vector = attribute_vector.cuda() if self.cuda else attribute_vector
            attribute_vector = attribute_vector[None, :]

            if self.cfg.baselines.single_autoencoder_baseline:
                pred_weights = weight_model(attribute_vector).squeeze()
            else:
                pred_weights = weight_model.predict(attribute_vector).squeeze()

            self.unseen_model.fc.weight.data[n, :] = pred_weights[:self.input_dim]
            self.unseen_model.fc.bias.data[n] = pred_weights[self.input_dim]
            self.ext_model.fc.weight.data[len(self.seen_classes) + n, :] = pred_weights[:self.input_dim]
            self.ext_model.fc.bias.data[len(self.seen_classes) + n] = pred_weights[self.input_dim]

        if daegnn:
            self.ref_weights = torch.cat(
                (self.ext_model.fc.weight.data, torch.unsqueeze(self.ext_model.fc.bias.data, 1)), 1
            ).unsqueeze(0)
            pred_weights = daegnn(self.ref_weights).squeeze()
            self.unseen_model.fc.weight.data[:, :] = pred_weights[len(self.seen_classes):, :self.input_dim]
            self.unseen_model.fc.bias.data[:] = pred_weights[len(self.seen_classes):, self.input_dim]
            self.ext_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim]
            self.ext_model.fc.bias.data[:] = pred_weights[:, self.input_dim]

        acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = self.val_gzsl()
        return acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_total = 0
        acc_per_class = []
        prediction_matrix = torch.zeros((len(target_classes), len(target_classes)))

        for n, i in enumerate(target_classes):
            idx = (test_label == i)
            if self.cfg.zsl.save_pred_matrix:
                for k, j in enumerate(target_classes):
                    prediction_matrix[n, k] = torch.sum(((predicted_label[idx]) == j)) / torch.sum(idx)
            acc = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
            acc_per_class.append(acc)
            acc_total += acc

        acc_total /= target_classes.size(0)
        acc_per_class = torch.stack(acc_per_class)

        return acc_total, acc_per_class, prediction_matrix

    def val_model(self, model, test_X, test_label, target_classes, calc_entropy = False):
        start = 0
        num_test = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        num_out = 0
        for layer in model.children():
            if hasattr(layer, 'out_features'):
                num_out = layer.out_features

        all_outputs = torch.Tensor(num_test, num_out)
        for i in range(0, num_test, self.batch_size):
            end = min(num_test, start + self.batch_size)
            if self.cuda:
                output = model(Variable(test_X[start:end].cuda()))
            else:
                output = model(Variable(test_X[start:end]))
            if calc_entropy:
                all_outputs[start:end] = output.data
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label, target_classes
        )

        if self.cfg.zsl.save_pred_matrix:
            torch.save(
                acc_per_class,
                self.cfg.output_root + '/predictions/percls_acc_'
                + self.cfg.dataset + self.cfg.zsl.image_embedding
                + '_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt'
            )
            torch.save(
                prediction_matrix,
                self.cfg.output_root + '/predictions/pred_matrix_'
                + self.cfg.dataset + self.cfg.zsl.image_embedding
                + '_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt'
            )

        if calc_entropy:
            from torch.distributions import Categorical
            sm = torch.nn.Softmax(dim=1)
            mean_entropy = Categorical(probs=sm(all_outputs)).entropy().mean()
            print("Mean entropy (log e) of output distributions over test samples: ", mean_entropy)
        return acc

    def val_gzsl(self):
        if self.cfg.zsl.norm_scale_heuristic:
            pred_weights = torch.cat(
                (self.unseen_model.fc.weight.data, torch.unsqueeze(self.unseen_model.fc.bias.data, 1)), 1)
            pred_weights = pred_weights / (10 * torch.norm(pred_weights, dim=1).mean())
            self.unseen_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim]
            self.unseen_model.fc.bias.data[:] = pred_weights[:, self.input_dim]
            self.ext_model.fc.weight.data[len(self.seen_classes):, :] = pred_weights[:, :self.input_dim]
            self.ext_model.fc.bias.data[len(self.seen_classes):] = pred_weights[:, self.input_dim]

        acc_gzsl = self.val_model(
            self.ext_model,
            torch.cat((self.test_seen_feature, self.test_unseen_feature), 0),
            torch.cat((utils.map_label(self.test_seen_label, self.seen_classes),
                       utils.map_label_extend(self.test_unseen_label, self.unseen_classes,
                                              self.seen_classes)), 0),
            torch.cat((utils.map_label(self.seen_classes, self.seen_classes),
                       utils.map_label_extend(self.unseen_classes, self.unseen_classes,
                                              self.seen_classes)), 0),
            calc_entropy=self.calc_entropy
        )

        acc_seen = self.val_model(
            self.ext_model,
            self.test_seen_feature,
            utils.map_label(self.test_seen_label, self.seen_classes),
            utils.map_label(self.seen_classes, self.seen_classes)
        )

        acc_unseen = self.val_model(
            self.ext_model,
            self.test_unseen_feature,
            utils.map_label_extend(self.test_unseen_label, self.unseen_classes, self.seen_classes),
            utils.map_label_extend(self.unseen_classes, self.unseen_classes, self.seen_classes)
        )
        H = utils.harmonic_mean(acc_seen, acc_unseen)

        # ZSL
        acc_unseen_zsl = self.val_model(
            self.unseen_model, self.test_unseen_feature,
            utils.map_label(self.test_unseen_label, self.unseen_classes),
            utils.map_label(self.unseen_classes, self.unseen_classes)
        )

        if self.cfg.baselines.daegnn:
            self.ref_weights = self.ref_weights.squeeze()
            self.ext_model.fc.weight.data[:, :] = self.ref_weights[:, :self.input_dim]
            self.ext_model.fc.bias.data[:] = self.ref_weights[:, self.input_dim]

        return acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

