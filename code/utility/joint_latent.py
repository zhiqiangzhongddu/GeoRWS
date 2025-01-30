import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import code.utility.utils as utils
from code.data_loader import GenericDataset
from code.utility.regressor import REGRESSOR
from code.baseline.cv.wDAEGNN.low_shot_learning.architectures.classifiers.weights_denoising_autoencoder import \
    WeightsDAE
from code.utility import model_bases


class Joint(REGRESSOR):
    def __init__(
            self, _train_x, _train_y, data_loader, _num_class, seed,
            _cuda = True, train_base = False, _lr = 0.001,
            _beta1 = 0.5, _num_epoch = 20, _batch_size = 100, _embed_dim = 1000, _num_layers = 3, cfg = None
    ):
        super().__init__(
            _train_x=_train_x, _train_y=_train_y, data_loader=data_loader, _num_class=_num_class, seed=seed,
            _cuda=_cuda, train_base=train_base, _lr=_lr, _beta1=_beta1, _num_epoch=_num_epoch,
            _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, cfg=cfg
        )
        self.cfg = cfg
        self.seed = seed
        self.lr = _lr
        self.beta1 = _beta1
        self.num_epoch = _num_epoch
        self.batch_size = _batch_size
        self.embed_dim = _embed_dim
        self.num_layers = _num_layers
        self.num_class = _num_class
        self.cuda = _cuda

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seen_classes = data_loader.seen_classes
        self.unseen_classes = data_loader.unseen_classes
        self.attribute = data_loader.attribute

        self.target_weights = self.target_weights.cuda() if self.cuda else self.target_weights

        self.weight_optimizer = None
        self.optimizer_weight_AE = None
        self.optimizer_attribute_AE = None
        self.Q = None
        self.R = None
        self.dae = None
        self.dae_loader = None
        self.dae_meta_batch_size = None
        self.dae_optimizer = None
        self.loader = None
        self.l1loss = None
        self.mse_loss = None

        attribute2weight_dataset, combined_seen_dataset = self.init_data()

        if self.cfg.cv.subspace_proj:
            self.init_subreg()

        self.AE_attribute = model_bases.AUTOENCODER(
            cfg=self.cfg,
            input_dim=self.attribute.size(1),
            embed_dim=self.embed_dim,
            num_layers=self.num_layers
        )
        self.AE_weight = model_bases.AUTOENCODER(
            cfg=self.cfg,
            input_dim=self.target_weights.size(1),
            embed_dim=self.embed_dim,
            num_layers=self.num_layers
        )

        if self.cfg.cv.single_autoencoder_baseline:
            self.reg_model = model_bases.AUTOENCODER(
                cfg=self.cfg,
                input_dim=self.attribute.size(1),
                embed_dim=self.embed_dim,
                output_dim=self.target_weights.size(1),
                num_layers=self.num_layers
            )
        else:
            self.reg_model = model_bases.JOINT_AUTOENCODER(
                cfg=self.cfg,
                autoencoder1=self.AE_attribute,
                autoencoder2=self.AE_weight
            )
        self.init_loss()

        if self.cfg.cv.daegnn:
            self.init_daegnn(
                attribute2weight_dataset=attribute2weight_dataset,
                combined_seen_dataset=combined_seen_dataset,
            )

        self.unseen_model = model_bases.LINEAR(
            self.test_seen_feature.size(1), len(self.unseen_classes)
        )
        self.ext_model = model_bases.LINEAR(
            self.test_seen_feature.size(1), len(self.seen_classes) + len(self.unseen_classes)
        )
        self.ext_model.fc.weight.data[:len(self.seen_classes), :] = self.target_weights[:, :-1]
        self.ext_model.fc.bias.data[:len(self.seen_classes)] = self.target_weights[:, -1]

        self.init_weight()
        if self.cuda:
            self.to_cuda(cfg=self.cfg)
        self.init_optimizer()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.acc_target, self.acc_zst_unseen = 0, 0
        self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = 0, 0, 0, 0, 0

    def init_data(self):
        if self.cfg.cv.class_reduction_ablation:
            perm = torch.randperm(len(self.seen_classes))
            assert self.cfg.cv.class_reduction_ablation in range(1, len(self.seen_classes) + 1)
            perm = perm[:self.cfg.cv.class_reduction_ablation]
            training_attributes = self.attribute[self.seen_classes][perm]
            training_weights = self.target_weights[perm]
        else:
            training_attributes, training_weights = None, None

        if self.cfg.cv.single_autoencoder_baseline:
            if self.cfg.cv.class_reduction_ablation:
                attribute2weight_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=training_attributes,
                    _target=training_weights,
                    cuda=self.cuda
                )
                self.loader = DataLoader(
                    attribute2weight_dataset, batch_size=self.batch_size, shuffle=True
                )
            else:
                attribute2weight_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=self.attribute[self.seen_classes],
                    _target=self.target_weights,
                    cuda=self.cuda
                )
                self.loader = DataLoader(
                    attribute2weight_dataset, batch_size=self.batch_size, shuffle=True
                )
            combined_seen_dataset = None
        else:
            attribute2weight_dataset = None
            if self.cfg.cv.class_reduction_ablation:
                combined_seen_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=training_attributes,
                    _target=training_weights,
                    cuda=self.cuda
                )
            else:
                combined_seen_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=self.attribute[self.seen_classes],
                    _target=self.target_weights,
                    cuda=self.cuda
                )
            placeholder_weights = torch.zeros(len(self.unseen_classes), self.target_weights.size(1))
            placeholder_weights = placeholder_weights.cuda() if self.cuda else placeholder_weights

            if self.cfg.cv.class_reduction_ablation:
                combined_full_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=torch.cat((training_attributes, self.attribute[self.unseen_classes])),
                    _target=torch.cat((training_weights, placeholder_weights), dim=0),
                    cuda=self.cuda
                )
            else:
                combined_full_dataset = GenericDataset(
                    cfg=self.cfg,
                    _input=torch.cat((self.attribute[self.seen_classes], self.attribute[self.unseen_classes])),
                    _target=torch.cat((self.target_weights, placeholder_weights), dim=0),
                    cuda=self.cuda
                )
            if self.cfg.include_unseen:
                self.loader = DataLoader(combined_full_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                self.loader = DataLoader(combined_seen_dataset, batch_size=self.batch_size, shuffle=True)

        return attribute2weight_dataset, combined_seen_dataset

    def init_loss(self):
        if self.cfg.cos_sim_loss:
            self.criterion = utils.cos_sim_loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1loss = nn.L1Loss(reduction='none')

    def init_subreg(self):
        # Sub. Reg. Baseline
        base_weights_mat = torch.cat(
            (self.reg_model.fc.weight.data, self.reg_model.fc.bias.data.unsqueeze(1)), 1
        )
        tr_base = torch.transpose(base_weights_mat, 0, 1)
        self.Q, self.R = torch.linalg.qr(tr_base, mode='reduced')

    def init_daegnn(self, attribute2weight_dataset, combined_seen_dataset):
        """
        initialise DAEGNN related parameters
        :param attribute2weight_dataset:
        :param combined_seen_dataset:
        """
        dae_num_features = 2049  # Number of features from ResNet-101, 512 in original implementation.
        self.dae_meta_batch_size = 4  # Taken from original implementation

        if self.cfg.cv.single_autoencoder_baseline:
            self.dae_loader = DataLoader(
                attribute2weight_dataset, batch_size=len(self.seen_classes), shuffle=False
            )
        else:
            self.dae_loader = DataLoader(
                combined_seen_dataset, batch_size=len(self.seen_classes), shuffle=False
            )

        self.dae = WeightsDAE({
            'gaussian_noise': 0.08,
            'comp_reconstruction_loss': True,
            'targets_as_input': False,
            'dae_type': 'RelationNetBasedGNN',
            'num_layers': 2,
            'num_features_input': dae_num_features,
            'num_features_output': 2 * dae_num_features,
            'num_features_hidden': 3 * dae_num_features,
            'update_dropout': 0.7,

            'nun_features_msg': 3 * dae_num_features,
            'aggregation_dropout': 0.7,
            'topK_neighbors': 10,
            'temperature': 5.0,
            'learn_temperature': False,
        })
        self.dae_optimizer = optim.Adam(
            self.dae.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=0.0
        )

    def init_weight(self):
        if self.reg_model:
            self.reg_model.apply(utils.weights_init)
        self.AE_attribute.apply(utils.weights_init)
        self.AE_weight.apply(utils.weights_init)

    def init_optimizer(self):
        self.optimizer_attribute_AE = optim.Adam(
            self.AE_attribute.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        self.optimizer_weight_AE = optim.Adam(
            self.AE_weight.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        if self.reg_model:
            self.weight_optimizer = optim.Adam(
                self.reg_model.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=0.0
            )

    def to_cuda(self, cfg):
        self.AE_attribute.cuda()
        self.AE_weight.cuda()
        if self.reg_model:
            self.reg_model.cuda()
        self.criterion.cuda()
        self.mse_loss.cuda()
        self.l1loss.cuda()
        if cfg.cv.daegnn:
            self.dae.cuda()
        self.ext_model.cuda()
        self.unseen_model.cuda()

    def fit_wdae_epoch(self, batch, comp_loss):
        self.reg_model.zero_grad()
        self.dae.zero_grad()

        if self.cfg.cv.single_autoencoder_baseline:
            attribute, weights = batch
            output = self.reg_model(attribute).detach()
            perm = torch.randperm(weights.size(0))
            weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

            num_idxs = weights.size(0) // self.dae_meta_batch_size
            for i in range(self.dae_meta_batch_size):
                idx = perm[i * num_idxs:(i + 1) * num_idxs]
                weights_input[i][idx] = output[idx]

            recon = self.dae(weights_input)
            loss = comp_loss(recon, weights_input).mean()
            loss.backward()
            self.dae_optimizer.step()
        else:
            attribute, weights = batch
            (attribute_from_attribute, attribute_from_weight,
             weight_from_weight, weight_from_attribute,
             latent_attribute, latent_weight) = self.reg_model(batch)
            perm = torch.randperm(weights.size(0))
            weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

            num_idxs = weights.size(0) // self.dae_meta_batch_size
            for i in range(self.dae_meta_batch_size):
                idx = perm[i * num_idxs:(i + 1) * num_idxs]
                weights_input[i][idx] = weight_from_attribute[idx]

            recon = self.dae(weights_input)
            loss = comp_loss(recon, weights_input).mean()
            loss.backward()
            self.dae_optimizer.step()

        return loss

    def fit_wdae(
            self,
            acc_target, acc_zst_unseen,
            run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen,
            run_best_H, run_best_unseen_zsl,
    ):
        print("Starting training of wDAE-GNN")
        comp_loss = nn.MSELoss(reduction='none')
        counter = 0
        breaking = False
        epoch_losses = []

        for epoch in range(self.num_epoch):
            epoch_loss = 0
            for _, batch in enumerate(self.dae_loader):
                loss = self.fit_wdae_epoch(batch=batch, comp_loss=comp_loss)
                epoch_loss += loss.data

            epoch_loss /= len(self.dae_loader)
            epoch_losses.append(epoch_loss)
            epoch_info = {"loss": epoch_loss}

            if self.cfg.early_stopping_slope:
                if epoch > 20:
                    threshold = 2 * 10e-4 if self.cfg.cos_sim_loss else 2 * 10e-7
                    slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(
                        torch.stack(epoch_losses)[-20:-10])) / 10.
                    if slope < threshold:
                        counter += 1
                        if counter == 5:
                            breaking = True
                    else:
                        counter = 0
                    epoch_info["slope"] = slope

            # Check down-stream performance (ZSL or GZSL) of weights predicted by current network state.
            # Note that performances seen here cannot be reported,
            # as we are implicitly assuming access to images during training to do this.
            if (not self.cfg.strict_eval) or (epoch + 1 == self.num_epoch) or breaking:
                self.reg_model.eval()
                self.dae.eval()
                if epoch + 1 == self.num_epoch or breaking:
                    self.calc_entropy = self.cfg.cv.calc_entropy

                val_out = self.pred_weights_and_val(weight_model=self.reg_model, daegnn=self.dae)
                self.reg_model.train()
                self.dae.train()

                if self.cfg.zst:
                    acc_target, acc_zst_unseen = val_out
                else:
                    acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                    epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                    epoch_info["H"] = H
                    epoch_info["acc_unseen_gzsl"] = acc_unseen
                    epoch_info["acc_seen_gzsl"] = acc_seen

                    # Save best performing downstream model
                    if H >= run_best_H:
                        print("New best GZSL based on H (seed):", H)
                        (run_best_acc_gzsl, run_best_acc_seen,
                         run_best_acc_unseen, run_best_H, run_best_unseen_zsl) \
                            = acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

            if breaking:
                print("Stopping early")
                break
            else:
                pass

        return (acc_target, acc_zst_unseen,
                run_best_acc_gzsl, run_best_acc_seen,
                run_best_acc_unseen, run_best_H, run_best_unseen_zsl)

    def fit_epoch(
            self, batch,
            epoch_loss,
            epoch_attribute_from_attribute_loss,
            epoch_attribute_from_weight_loss,
            epoch_weight_from_weight_loss,
            epoch_weight_from_attribute_loss,
    ):
        # Create mask to remove loss from weight prediction from unseen class attributes
        mask = torch.where(torch.sum(torch.abs(batch[1]), dim=-1) > 0., 1., 0.)[:, None]
        if self.cuda:
            mask = mask.cuda()
        mask_sum = torch.clamp(mask.sum(), min=1.)

        self.reg_model.zero_grad()

        if self.cfg.cv.single_autoencoder_baseline:
            attribute, weights = batch
            output = self.reg_model(attribute)
            loss = self.criterion(output, weights)
            loss = loss.mean()

            if self.cfg.cv.subspace_proj:
                mut = output @ self.Q
                mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                proj_weights = mutnorm @ self.Q.T
                proj_weights = proj_weights.squeeze()
                subspace_proj_loss = 0.001 * torch.norm(output - proj_weights, dim=-1).mean()
                loss += subspace_proj_loss

        else:
            (attribute_from_attribute, attribute_from_weight,
             weight_from_weight, weight_from_attribute,
             latent_attribute, latent_weight) = self.reg_model(batch)

            attribute_from_attribute_loss = self.criterion(attribute_from_attribute, batch[0]).mean()
            attribute_from_weight_loss = (self.criterion(attribute_from_weight, batch[0]) * mask).sum(
                0).mean() / mask_sum
            if self.cfg.cv.single_modal_ablation:
                attribute_from_weight_loss = 0 * attribute_from_weight_loss
            weight_from_weight_loss = ((self.criterion(weight_from_weight, batch[1]) * mask).sum(0).mean()
                                       / mask_sum)
            weight_from_attribute_loss = ((self.criterion(weight_from_attribute, batch[1]) * mask).sum(0).mean()
                                          / mask_sum)

            loss = (attribute_from_attribute_loss + attribute_from_weight_loss
                    + weight_from_weight_loss + weight_from_attribute_loss)

            epoch_attribute_from_attribute_loss += attribute_from_attribute_loss.data
            epoch_attribute_from_weight_loss += attribute_from_weight_loss.data
            epoch_weight_from_weight_loss += weight_from_weight_loss.data
            epoch_weight_from_attribute_loss += weight_from_attribute_loss.data

            if self.cfg.cv.subspace_proj:
                mut = weight_from_attribute @ self.Q
                mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                proj_weights = mutnorm @ self.Q.T
                proj_weights = proj_weights.squeeze()
                subspace_proj_loss = 0.001 * torch.norm(weight_from_attribute - proj_weights)
                loss += subspace_proj_loss

        epoch_loss += loss.data

        loss.backward()
        self.weight_optimizer.step()

        return (epoch_loss,
                epoch_attribute_from_attribute_loss, epoch_attribute_from_weight_loss,
                epoch_weight_from_weight_loss, epoch_weight_from_attribute_loss)

    def fit(self):
        """
        Fit
        """
        acc_target, acc_zst_unseen = 0, 0
        run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl \
            = 0, 0, 0, 0, 0

        counter = 0
        breaking = False
        epoch_losses = []

        for epoch in range(self.num_epoch):
            epoch_loss = 0
            epoch_attribute_from_attribute_loss = 0
            epoch_attribute_from_weight_loss = 0
            epoch_weight_from_weight_loss = 0
            epoch_weight_from_attribute_loss = 0

            for i_batch, batch in enumerate(self.loader):
                (epoch_loss,
                 epoch_attribute_from_attribute_loss, epoch_attribute_from_weight_loss,
                 epoch_weight_from_weight_loss, epoch_weight_from_attribute_loss) = self.fit_epoch(
                    batch=batch,
                    epoch_loss=epoch_loss,
                    epoch_attribute_from_attribute_loss=epoch_attribute_from_attribute_loss,
                    epoch_attribute_from_weight_loss=epoch_attribute_from_weight_loss,
                    epoch_weight_from_weight_loss=epoch_weight_from_weight_loss,
                    epoch_weight_from_attribute_loss=epoch_weight_from_attribute_loss,
                )

            epoch_loss /= len(self.loader)
            epoch_losses.append(epoch_loss)

            if self.cfg.cv.single_autoencoder_baseline:
                epoch_info = {"loss": epoch_loss}
            else:
                epoch_info = {
                    "loss": epoch_loss,
                    "attribute_from_attribute_loss": epoch_attribute_from_attribute_loss,
                    "attribute_from_weight_loss": epoch_attribute_from_weight_loss,
                    "weight_from_weight_loss": epoch_weight_from_weight_loss,
                    "weight_from_attribute_loss": epoch_weight_from_attribute_loss
                }

            if self.cfg.early_stopping_slope:
                if epoch > 20:
                    threshold = 2 * 10e-4 if self.cfg.cos_sim_loss else 2 * 10e-7
                    slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(
                        torch.stack(epoch_losses)[-20:-10])) / 10.
                    if slope < threshold:
                        counter += 1
                        if counter == 5:
                            breaking = True
                    else:
                        counter = 0
                    epoch_info["slope"] = slope

            # Check down-stream performance (ZSL or GZSL) of weights predicted by current network state.
            # Note that performances seen here cannot be reported,
            # as we are implicitly assuming access to images during training to do this.
            if ((not self.cfg.strict_eval) or (epoch + 1 == self.num_epoch) or breaking) and not self.cfg.cv.daegnn:
                self.reg_model.eval()
                if epoch + 1 == self.num_epoch or breaking:
                    self.calc_entropy = self.cfg.cv.calc_entropy

                val_out = self.pred_weights_and_val(weight_model=self.reg_model)
                self.reg_model.train()

                if self.cfg.zst:
                    acc_target, acc_zst_unseen = val_out
                else:
                    acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                    epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                    epoch_info["H"] = H
                    epoch_info["acc_unseen_gzsl"] = acc_unseen
                    epoch_info["acc_seen_gzsl"] = acc_seen

                    # Save best performing downstream model
                    if H >= run_best_H:
                        (run_best_acc_gzsl, run_best_acc_seen,
                         run_best_acc_unseen, run_best_H, run_best_unseen_zsl) = (
                            acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl)

            if breaking:
                print("Stopping early (slope criterion)")
                break
            else:
                pass

        if self.cfg.cv.daegnn:
            (acc_target, acc_zst_unseen,
             run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen,
             run_best_H, run_best_unseen_zsl) = self.fit_wdae(
                acc_target=acc_target, acc_zst_unseen=acc_zst_unseen,
                run_best_acc_gzsl=run_best_acc_gzsl, run_best_acc_seen=run_best_acc_seen,
                run_best_acc_unseen=run_best_acc_unseen, run_best_H=run_best_H,
                run_best_unseen_zsl=run_best_unseen_zsl,
            )
        else:
            pass

        if self.cfg.zst:
            self.acc_target, self.acc_zst_unseen = acc_target, acc_zst_unseen
        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl \
                = (run_best_acc_gzsl, run_best_acc_seen,
                   run_best_acc_unseen, run_best_H, run_best_unseen_zsl)
