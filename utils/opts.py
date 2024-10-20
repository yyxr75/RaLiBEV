# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # ----------------------
        # dataset file names
        # ----------------------
        self.parser.add_argument('--training_data', default='data/datasets/oxford_dataset/dataset_filename/train.txt')
        self.parser.add_argument('--validation_data', default='data/datasets/oxford_dataset/dataset_filename/val.txt')
        self.parser.add_argument('--test_data', default='data/datasets/oxford_dataset/dataset_filename/eval.txt')
        self.parser.add_argument('--model_path', default='outputs/logs/DQMITBF/epoch191-train_0.088-val_0.896_ckpt.pth')
        self.parser.add_argument('--resume', type=int, default=0)

        # log direction
        #  training log director
        self.parser.add_argument('--output_dir', default='./outputs/logs')
        self.parser.add_argument('--experiment_name', default='DQMITBF')
        #  evaluation log director
        self.parser.add_argument('--map_out', default='./outputs/logs')
        self.parser.add_argument('--exp_name', default='DQMITBF')

        # ----------------------
        # using multi scale output for both training and validation and test
        # ----------------------
        self.parser.add_argument('--using_multi_scale', type=int, default=1,
                                 help='whether to using multi scale')
        # ----------------------
        # training params
        # ----------------------
        self.parser.add_argument('--batch_size', type=int, default=2,
                                 help='input batch size')
        self.parser.add_argument('--data_shape', type=int, nargs='+', default=[320, 320],
                                 help='the input data shape')
        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 help='start epoch')
        self.parser.add_argument('--end_epoch', type=int, default=200)
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='how many cpus')
        self.parser.add_argument('--cuda', type=int, default=1,
                                 help='1 means this machine has gpu')

        self.parser.add_argument('--rotate_angle', type=int, default=0,
                                 help='0 means ban random rotate'
                                 '1 means allow random rotate')
        # ----------------------
        # network architecture
        # ----------------------
        self.parser.add_argument('--modality', default='fusion',
                                 help='whether fuse radar and lidar data or use lidar only'
                                 'fusion: fuse lidar and radar'
                                 'lidar-only: use lidar only'
                                 'radar-only: use radar only')
        self.parser.add_argument('--fusion_arch', default='Dense_Query_Map-based_Interactive_Transformer_for_BEV_Fusion_(DQMITBF)',
                                 help='where to cat the data'
                                 'preCat: concatenate in raw spatial dimension'
                                 'Dense_Query_Map-based_Interactive_Transformer_for_BEV_Fusion_(DQMITBF): cross attention use random initialized variable query embedding as q, cat two data of radar/lidar as k/v, lidar/radar as k/v'
                                 )
        self.parser.add_argument('--fusion_loss_arch', default='fusion_loss_base')
        # ----------------------
        # loss parameters
        # ----------------------
        self.parser.add_argument('--binNum', type=int, default=8)
        # ----------------------
        # debug parameters
        # ----------------------
        self.parser.add_argument('--showHeatmap', type=int, default=0,
                                 help='whether to show heatmap')
        self.parser.add_argument('--chooseLoss', type=int, default=0)
        self.parser.add_argument('--use_hungarian_match', type=int, default=0,
                                 help='whether to use hungarian match for label assignment')

        # ----------------------
        # training parameters
        # ----------------------
        # learning rate scheduler
        self.parser.add_argument('--lr_scheduler', default='warmup_multistep')
        self.parser.add_argument('--base_lr', type=float, default=0.01,
                                 help='base learning rate')

        self.parser.add_argument('--min_lr_ratio', type=float, default=0.05)
        self.parser.add_argument('--warmup_epochs', type=int, default=2)
        self.parser.add_argument('--warmup_lr', type=float, default=0)

        # optimizer parameters
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                 help='base learning rate')

        # multi gpu training
        self.parser.add_argument('--distributed', type=int, default=0,
                                 help='whether use multi gpu training: True, False')
        self.parser.add_argument('--from_distributed', type=bool, default=True,
                                 help='whether read model from distributed training')
        self.parser.add_argument('--main_gpuid', type=int, default=0,
                                 help='main gpu id: 0, 1, 2 ...')
        self.parser.add_argument('--local_rank', type=int, help="local gpu id")
        self.parser.add_argument('--world_size', type=int, help="num of processes")

        # evaluator parameters
        self.parser.add_argument('--mAP_type', default='pycoco',
                                 help='choose mAP calculator: '
                                      'norm0.5'
                                      'pycoco')

        self.parser.add_argument('--heatmap_thresh', type=float, default=0.1)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt
