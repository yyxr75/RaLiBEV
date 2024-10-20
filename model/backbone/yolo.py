# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from model.backbone.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.LeakyReLU(0.1),
    )

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class YoloDecoder(nn.Module):
    def __init__(self, in_channel):
        super(YoloDecoder, self).__init__()
        self.backbone = darknet53(None, in_channel)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = make_three_conv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = conv2d(256,128,1)
        self.make_five_conv2    = make_five_conv([128, 256],256)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)


    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)
        return P3, P4, P5

class YoloDecoder_2branch(nn.Module):
    def __init__(self, in_channel_lidar, in_channel_radar):
        super(YoloDecoder_2branch, self).__init__()
        self.backbone_lidar = darknet53(None, in_channel_lidar)
        self.backbone_radar = darknet53(None, in_channel_radar)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = make_three_conv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = conv2d(256,128,1)
        self.make_five_conv2    = make_five_conv([128, 256],256)
        self.make_five_conv2_channel_downsample    = make_five_conv([128, 256],512)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)
        self.make_five_conv3_channel_downsample    = make_five_conv([256, 512],1024)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)
        self.make_five_conv4_channel_downsample    = make_five_conv([512, 1024],2048)


    def forward(self, lidar, radar):

        lidar_x2, lidar_x1, lidar_x0 = self.backbone_lidar(lidar)
        radar_x2, radar_x1, radar_x0 = self.backbone_radar(radar)

        lidar_P5 = self.conv1(lidar_x0)
        radar_P5 = self.conv1(radar_x0)
        lidar_P5 = self.SPP(lidar_P5)
        radar_P5 = self.SPP(radar_P5)
        lidar_P5 = self.conv2(lidar_P5)
        radar_P5 = self.conv2(radar_P5)

        lidar_P5_upsample = self.upsample1(lidar_P5)
        radar_P5_upsample = self.upsample1(radar_P5)
        lidar_P4 = self.conv_for_P4(lidar_x1)
        radar_P4 = self.conv_for_P4(radar_x1)
        lidar_P4 = torch.cat([lidar_P4,lidar_P5_upsample],axis=1)
        radar_P4 = torch.cat([radar_P4,radar_P5_upsample],axis=1)
        P4 = torch.cat([lidar_P4, radar_P4],axis=1)
        lidar_P4 = self.make_five_conv1(lidar_P4)
        radar_P4 = self.make_five_conv1(radar_P4)

        lidar_P4_upsample = self.upsample2(lidar_P4)
        radar_P4_upsample = self.upsample2(radar_P4)
        lidar_P3 = self.conv_for_P3(lidar_x2)
        radar_P3 = self.conv_for_P3(radar_x2)
        lidar_P3 = torch.cat([lidar_P3,lidar_P4_upsample],axis=1)
        radar_P3 = torch.cat([radar_P3,radar_P4_upsample],axis=1)
        P3 = torch.cat([lidar_P3,radar_P3],axis=1)
        lidar_P3 = self.make_five_conv2(lidar_P3)
        radar_P3 = self.make_five_conv2(radar_P3)
        P3 = self.make_five_conv2_channel_downsample(P3)

        lidar_P3_downsample = self.down_sample1(lidar_P3)
        radar_P3_downsample = self.down_sample1(radar_P3)
        lidar_P4 = torch.cat([lidar_P3_downsample,lidar_P4],axis=1)
        radar_P4 = torch.cat([radar_P3_downsample,radar_P4],axis=1)
        P4 = torch.cat([lidar_P4,radar_P4],axis=1)
        lidar_P4 = self.make_five_conv3(lidar_P4)
        radar_P4 = self.make_five_conv3(radar_P4)
        P4 = self.make_five_conv3_channel_downsample(P4)

        lidar_P4_downsample = self.down_sample2(lidar_P4)
        radar_P4_downsample = self.down_sample2(radar_P4)
        lidar_P5 = torch.cat([lidar_P4_downsample,lidar_P5],axis=1)
        radar_P5 = torch.cat([radar_P4_downsample,radar_P5],axis=1)
        P5 = torch.cat([lidar_P5,radar_P5],axis=1)
        P5 = self.make_five_conv4_channel_downsample(P5)
        return P3, P4, P5
