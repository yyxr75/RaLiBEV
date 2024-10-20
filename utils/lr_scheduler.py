# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
import torch
from typing import List
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int], # epoch number
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_epoch: int = 2,
        iters_per_epoch: int = 220,
        warmup_method: str = "linear",
        last_epoch: int = -1, # from father object
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = [i * iters_per_epoch for i in milestones]
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_epoch*iters_per_epoch
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        lr = [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
        return lr

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class LRScheduler:
    def __init__(self, name, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Supported lr schedulers: [cos, warmcos, multistep]

        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - cos: None
                - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
                - multistep: [milestones (epochs), gamma (default 0.1)]
        """

        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)

        self.lr_func = self._get_lr_func(name)

    def update_lr(self, iters):
        return self.lr_func(iters)

    def _get_lr_func(self, name):
        if name == "cos":  # cosine lr schedule
            lr_func = partial(cos_lr, self.lr, self.total_iters)
        elif name == "warmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
            lr_func = partial(
                warm_cos_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        elif name == "yoloxwarmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
            lr_func = partial(
                yolox_warm_cos_lr,
                self.lr,
                min_lr_ratio,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        elif name == "yoloxsemiwarmcos":
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            normal_iters = self.iters_per_epoch * self.semi_epoch
            semi_iters = self.iters_per_epoch_semi * (
                self.total_epochs - self.semi_epoch - self.no_aug_epochs
            )
            lr_func = partial(
                yolox_semi_warm_cos_lr,
                self.lr,
                min_lr_ratio,
                warmup_lr_start,
                self.total_iters,
                normal_iters,
                no_aug_iters,
                warmup_total_iters,
                semi_iters,
                self.iters_per_epoch,
                self.iters_per_epoch_semi,
            )
        elif name == "multistep":  # stepwise lr schedule
            milestones = [
                int(self.total_iters * milestone / self.total_epochs)
                for milestone in self.milestones
            ]
            gamma = getattr(self, "gamma", 0.1)
            lr_func = partial(multistep_lr, self.lr, milestones, gamma)
        elif name == "warmup_multistep":
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            milestones = [
                int(self.total_iters * milestone / self.total_epochs)
                for milestone in self.milestones
            ]
            gamma = getattr(self, "gamma", 0.1)
            lr_func = partial(warmup_multistep_lr, self.lr, warmup_total_iters, warmup_lr_start, milestones, gamma)
        else:
            raise ValueError("Scheduler version {} not supported.".format(name))

        return lr_func


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr


def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr


def yolox_semi_warm_cos_lr(
    lr,
    min_lr_ratio,
    warmup_lr_start,
    total_iters,
    normal_iters,
    no_aug_iters,
    warmup_total_iters,
    semi_iters,
    iters_per_epoch,
    iters_per_epoch_semi,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= normal_iters + semi_iters:
        lr = min_lr
    elif iters <= normal_iters:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (
                    normal_iters
                    - warmup_total_iters
                    + (iters - normal_iters)
                    * iters_per_epoch
                    * 1.0
                    / iters_per_epoch_semi
                )
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    return lr


def multistep_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate"""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr


def warmup_multistep_lr(lr, warmup_total_iters, warmup_lr_start, milestones, gamma, iters):
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    else:
        for milestone in milestones:
            lr *= gamma if iters >= milestone else 1.0
    return lr
