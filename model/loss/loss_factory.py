# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .fusion_loss_base import fusion_loss_base

loss_factory = {
    'fusion_loss_base': fusion_loss_base,
}


def get_fusionloss(opts):
    fusion_loss = loss_factory[opts.fusion_loss_arch]
    fusion_loss = fusion_loss(opts)
    return fusion_loss

