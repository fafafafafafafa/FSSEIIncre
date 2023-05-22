import warnings
import argparse
from copy import deepcopy
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import functional as F

from approachs.incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from loss_funcs.triplet_losses import TripletLoss


class Appr(Inc_Learning_Appr):
    """Class implementing the End-to-end Incremental Learning (EEIL) approach described in
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Original code available at https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code from https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=40, noise_grad=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad

        self._train_epoch = 0
        self._finetuning_balanced = None

        # EEIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: EEIL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = argparse.ArgumentParser()
        # Added trade-off between the terms of Eq. 1 -- L = L_C + lamb * L_D
        parser.add_argument('--lamb', default=1.0, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 6: "Based on our empirical results, we set T to 2 for all our experiments"
        parser.add_argument('--T', default=2.0, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr-finetuning-factor', default=0.01, type=float, required=False,
                            help='Finetuning learning rate factor (default=%(default)s)')
        # Number of epochs for balanced training
        parser.add_argument('--nepochs-finetuning', default=1, type=int, required=False,
                            help='Number of epochs for balanced training (default=%(default)s)')
        # the addition of noise to the gradients
        parser.add_argument('--noise-grad', action='store_true',
                            help='Add noise to gradients (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0:
            # t == 0时,
            super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            loader = self._train_unbalanced(t, trn_loader, val_loader)
            self._train_balanced(t, trn_loader, val_loader)

        # 更新保存 见过类 的少量样本
        self.exemplars_dataset.collect_exemplars(self.model, loader, None)

    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _get_train_loader(self, trn_loader, balanced=False):
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            # 新类和旧类的数量相同
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])
        ds = trn_dataset + exemplars_ds
        return DataLoader(ds,
                          batch_size=trn_loader.batch_size,
                          shuffle=True,
                          num_workers=trn_loader.num_workers,
                          pin_memory=trn_loader.pin_memory)

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, False)
        super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, t, trn_loader, val_loader):
        self._finetuning_balanced = True
        self._train_epoch = 0

        orign_nepochs = self.nepochs
        orign_lr = self.lr
        self.nepochs = self.nepochs_finetuning
        self.lr = self.lr * self.lr_finetuning_factor

        loader = self._get_train_loader(trn_loader, True)
        super().train_loop(t, loader, val_loader)

        self.nepochs = orign_nepochs
        self.lr = orign_lr

    def train_epoch(self, t, trn_loader):
        self.model.train()
        running_loss = 0
        for batch, (images, targets) in enumerate(trn_loader, 1):
            images = images.to(self.device)
            outputs_old = None
            if t > 0 and self._finetuning_balanced:
                outputs_old = self.model_old(images)
            outputs, feats = self.model(images, return_features=True)
            loss = self.criterion(t, outputs, targets.long().to(self.device), outputs_old)
            #if self._finetuning_balanced :
            #    tri_loss = TripletLoss()(feats, torch.unsqueeze(targets, 1).long().to(self.device), margin=2)
            #    loss += tri_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().item()
            if batch == len(trn_loader):
                print('train_loss:', running_loss/batch)
        # _train_epoch 用于添加梯度噪声, 目前没用上
        self._train_epoch += 1

    def criterion(self, t, outputs, targets, outputs_old=None):
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, 1), targets)
        # loss = torch.nn.CrossEntropyLoss()(torch.cat(outputs, 1), targets)
        if outputs_old is not None:
            # 当 trian_balabced 时, 考虑当前head; 否则, 只考虑之前的
            last_head_idx = t if self._finetuning_balanced else (t-1)
            for i in range(last_head_idx):
                # loss bce
                loss += torch.nn.functional.binary_cross_entropy(F.softmax(outputs[i]/self.T, dim=1),
                                                                 F.softmax(outputs_old[i]/self.T, dim=1))
        return loss









