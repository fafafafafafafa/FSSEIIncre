import random
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda

from datasets.exemplars_dataset import ExemplarsDataset
from networks.network import LLL_Net


class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform):
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(model)

        if trn_loader.batch_size is None:
            batch_size = trn_loader.batch_sampler.true_batch_size
        else:
            batch_size = trn_loader.batch_size

        with override_dataset_transform(trn_loader.dataset, transform) as ds_for_selection:
            # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
            sel_loader = DataLoader(ds_for_selection, batch_size=batch_size, shuffle=False,
                                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            selected_indices = self._select_indices(model, sel_loader, exemplars_per_class, transform)
        with override_dataset_transform(trn_loader.dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
            x, y = zip(*(ds_for_raw[idx] for idx in selected_indices))
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y

    def _exemplars_per_class_num(self, model: LLL_Net):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        num_cls = sum(model.task_cls)
        result = []
        labels = self._get_labels(sel_loader)
        for curr_cls in range(num_cls):
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where(labels == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # select the exemplars randomly
            result.extend(random.sample(list(cls_ind), exemplars_per_class))
        return result

    def _get_labels(self, sel_loader):
        if hasattr(sel_loader.dataset, 'Y'):  # BaseDataset, MemoryDataset
            labels = np.asarray(sel_loader.dataset.Y)
        elif isinstance(sel_loader.dataset, ConcatDataset):
            labels = []
            for ds in sel_loader.dataset.datasets:
                labels.extend(ds.Y)
            labels = np.array(labels)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
        return labels


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        model_device = next(model.parameters()).device
        # extract features
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                features = model(images.to(model_device), return_features=True)[1]
                features = features/features.norm(dim=1).view(-1, 1)

                extracted_features.append(features)
                extracted_targets.extend(targets)

        extracted_features = (torch.cat(extracted_features, dim=0)).cpu()
        extracted_targets = np.array(extracted_targets)

        # iterate class
        result = []
        for cur_cls in np.unique(extracted_targets):

            cur_cls_ind = np.where(extracted_targets == cur_cls)[0]
            assert (len(cur_cls_ind) > 0), "No samples to choose from for class {:d}".format(cur_cls_ind)
            if len(cur_cls_ind) <= exemplars_per_class:
                cls_sample = len(cur_cls_ind)
            else:
                cls_sample = exemplars_per_class
            # get mean features
            cur_cls_feature_mean = extracted_features[cur_cls_ind].mean(0)
            # select cur cls examples
            selected_features = []
            selected_ind = []
            for k in range(cls_sample):
                sum_selected_features = 0
                for j in selected_features:
                    sum_selected_features += j/(k+1)
                min_dist = np.inf
                # choose the closest to the mean of the current class
                for ind in cur_cls_ind:
                    if ind not in selected_ind:
                        feature = extracted_features[ind]
                        dist = torch.norm(cur_cls_feature_mean-feature/(k+1)-sum_selected_features)
                        if dist < min_dist:
                            min_dist = dist
                            min_feature = feature
                            min_ind = ind
                selected_features.append(min_feature)
                selected_ind.append(min_ind)
            result.extend(selected_ind)
        return result


def dataset_transforms(dataset, transform_to_change):
    if isinstance(dataset, ConcatDataset):
        r = []
        for ds in dataset.datasets:
            r += dataset_transforms(ds, transform_to_change)
        return r
    else:
        old_transform = dataset.transform_x
        dataset.transform_x = transform_to_change
        return [(dataset, old_transform)]


@contextmanager
def override_dataset_transform(dataset, transform):
    try:
        datasets_with_orig_transform = dataset_transforms(dataset, transform)
        yield dataset
    finally:
        # get bac original transformations
        for ds, orig_transform in datasets_with_orig_transform:
            ds.transform = orig_transform
